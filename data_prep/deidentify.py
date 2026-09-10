# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""De-identify a SHEBA fold for public release.

Copies the study series to a new tree and de-identifies every DICOM in place,
then verifies the result. Source is never modified.

Three things this does that a naive pass does not:

1. **Promotes the Philips private rescale to standard tags.** On the Ingenia
   Ambition S the intensity calibration lives only in ``(2005,140A)`` and
   ``(2005,1409)``; the standard ``(0028,1053)``/``(0028,1052)`` are absent.
   Stripping private tags without promoting them first silently destroys the
   calibration, leaving pixel values that cannot be converted to real
   intensities.

2. **Removes private tags.** They are not merely vendor noise here -- the
   ELSCINT1 block ``(07A3,104C)`` was found to contain patient identifier
   fragments. dicognito does not remove private tags on its own.

3. **Verifies.** Every written file is re-read and checked against the source
   identifiers. A single residual match fails the run rather than producing a
   quietly unsafe archive.

Usage:
    python -m data_prep.deidentify --src <fold>/original --out <dest>
    python -m data_prep.deidentify --src ... --out ... --cases 1 2   # subset
"""

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pydicom
from pydicom.tag import Tag
from dicognito.anonymizer import Anonymizer

#: Series carried into the release -- the contrasts the study uses. Everything
#: else in `original/` (DTI, the *_TFE series) is unrelated patient imaging and
#: is deliberately left behind.
SERIES = ['T1_HR', 'T2_HR', 'FLAIR_HR',
          'T1_LR', 'T2_LR', 'FLAIR_LR',
          'BICUBIC_T1_LR', 'BICUBIC_T2_LR', 'BICUBIC_FLAIR_LR']

PHILIPS_SLOPE = (0x2005, 0x140A)
PHILIPS_INTERCEPT = (0x2005, 0x1409)
STD_SLOPE = (0x0028, 0x1053)
STD_INTERCEPT = (0x0028, 0x1052)

#: Tags read to build the set of strings that must not survive into the output.
TEXT_IDENTIFIER_TAGS = [(0x0010, 0x0010), (0x0010, 0x0020), (0x0008, 0x0050),
                        (0x0020, 0x0010), (0x0008, 0x0080), (0x0008, 0x0081),
                        (0x0008, 0x0090), (0x0008, 0x1010), (0x0018, 0x1000)]

#: Checked whole, never in pieces. A UID is a long dot-separated digit string, so
#: its 4-digit chunks collide by chance with the freshly generated replacements --
#: which reads as a leak when it is not.
UID_TAGS = [(0x0020, 0x000D), (0x0020, 0x000E),
            (0x0008, 0x0018), (0x0020, 0x0052)]

#: Device identifiers dicognito leaves in place. DeviceSerialNumber is on the
#: PS3.15 Annex E list and identifies the individual scanner -- it was found
#: surviving a first pass because the station name embeds the same serial.
DEVICE_TAGS = [(0x0018, 0x1000),  # DeviceSerialNumber
               (0x0018, 0x1030),  # ProtocolName -- free text
               (0x0040, 0x0275),  # RequestAttributesSequence
               (0x0032, 0x1060),  # RequestedProcedureDescription
               (0x0038, 0x0010),  # AdmissionID
               (0x0008, 0x1048),  # PhysiciansOfRecord
               (0x0008, 0x1050),  # PerformingPhysicianName
               (0x0008, 0x1070)]  # OperatorsName

#: Patient attributes dicognito preserves by design. None is needed for image
#: restoration, and with a cohort this small the combination of sex, age and
#: weight is a quasi-identifier. PatientSex is Type 2 -- it must remain present,
#: so it is emptied rather than deleted; the rest are Type 3 and are removed.
PATIENT_ATTR_DELETE = [(0x0010, 0x1010),  # PatientAge
                       (0x0010, 0x1030),  # PatientWeight
                       (0x0010, 0x1020),  # PatientSize
                       (0x0010, 0x2160),  # EthnicGroup
                       (0x0010, 0x21B0),  # AdditionalPatientHistory
                       (0x0010, 0x4000)]  # PatientComments
PATIENT_ATTR_EMPTY = [(0x0010, 0x0040)]   # PatientSex (Type 2)

#: Legitimately preserved and non-identifying. Excluded from the residual scan
#: so that e.g. Manufacturer='PHILIPS' does not trip on a station name that
#: happens to contain the vendor string.
VERIFY_SKIP = {Tag(0x0008, 0x0070),  # Manufacturer
               Tag(0x0008, 0x1090),  # ManufacturerModelName
               Tag(0x0018, 0x1020),  # SoftwareVersions
               Tag(0x0008, 0x0060),  # Modality
               Tag(0x0018, 0x0087)}  # MagneticFieldStrength


def identifier_fragments(ds):
    """Substrings that would re-identify the subject if they survived.

    Vendor strings are excluded: the station name embeds the manufacturer, which
    is legitimately preserved in ``Manufacturer`` and in Philips LUT
    explanations, and would otherwise trip the scan on every file.
    """
    vendor = set()
    for tag in ((0x0008, 0x0070), (0x0008, 0x1090)):
        if tag in ds:
            for part in re.split(r'[\^\s,._-]+', str(ds[tag].value).strip()):
                if part:
                    vendor.add(part.upper())

    out = set()
    for tag in TEXT_IDENTIFIER_TAGS:
        if tag not in ds:
            continue
        for part in re.split(r'[\^\s,._-]+', str(ds[tag].value).strip()):
            if len(part) >= 4 and part.upper() not in vendor:
                out.add(part.upper())
    uids = {str(ds[t].value).strip() for t in UID_TAGS if t in ds}
    return out, {u for u in uids if u}


def promote_rescale(ds):
    """Copy the Philips private rescale into the standard tags.

    Returns True if a promotion happened. Without this, removing private tags
    discards the only copy of the calibration.
    """
    if STD_SLOPE in ds and STD_INTERCEPT in ds:
        return False
    if PHILIPS_SLOPE not in ds:
        return False
    ds.RescaleSlope = float(ds[PHILIPS_SLOPE].value)
    ds.RescaleIntercept = float(ds[PHILIPS_INTERCEPT].value) if PHILIPS_INTERCEPT in ds else 0.0
    ds.RescaleType = 'US'
    return True


def deidentify_file(path, dest, anonymizer, stats):
    ds = pydicom.dcmread(str(path), force=True)
    fragments, uids = identifier_fragments(ds)

    if promote_rescale(ds):
        stats['rescale promoted'] += 1

    anonymizer.anonymize(ds)
    ds.remove_private_tags()
    for tag in DEVICE_TAGS + PATIENT_ATTR_DELETE:
        if tag in ds:
            del ds[tag]
    for tag in PATIENT_ATTR_EMPTY:
        if tag in ds:
            ds[tag].value = ''
    ds.PatientIdentityRemoved = 'YES'
    ds.DeidentificationMethod = 'dicognito PS3.15-E; private tags removed; rescale promoted'

    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(dest), enforce_file_format=True)
    return fragments, uids


def verify_file(dest, checks, stats):
    """Re-read a written file and fail on any residual identifier."""
    ds = pydicom.dcmread(str(dest), force=True)
    problems = []

    if sum(1 for e in ds if e.tag.is_private):
        problems.append('private tags present')
    if STD_SLOPE not in ds:
        problems.append('rescale slope missing')
    for tag in DEVICE_TAGS + PATIENT_ATTR_DELETE:
        if tag in ds:
            problems.append(f'tag {tag} present')
    for tag in PATIENT_ATTR_EMPTY:
        if tag in ds and str(ds[tag].value).strip():
            problems.append(f'tag {tag} populated')

    # Text identifiers are searched in text fields only. Searching them inside
    # UIDs produces false positives: a 5-digit serial number turns up by chance
    # inside a freshly generated 40-digit UID roughly once in 3000 files.
    blob, uid_blob = [], []
    for e in ds:
        if e.tag in VERIFY_SKIP:
            continue
        try:
            value = str(e.value).upper()
        except Exception:
            continue
        if e.VR == 'UI':
            uid_blob.append(value)
        elif e.VR in ('PN', 'LO', 'SH', 'ST', 'LT', 'UT', 'AS', 'DA', 'TM', 'CS'):
            blob.append(value)
    fragments, uids = checks
    joined = ' '.join(blob)
    for frag in fragments:
        if frag in joined:
            problems.append(f'identifier fragment {frag[:3]}... survived')
    uid_joined = ' '.join(uid_blob)
    for uid in uids:
        if uid in uid_joined:
            problems.append('a source UID survived verbatim')
    if problems:
        stats['FAILED'] += 1
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    ap.add_argument('--src', required=True, help='the fold\'s original/ directory')
    ap.add_argument('--out', required=True, help='destination (created; must not be inside src)')
    ap.add_argument('--cases', nargs='+', help='subset of case directories')
    ap.add_argument('--series', nargs='+', default=SERIES)
    ap.add_argument('--verify-only', action='store_true',
                    help='re-check an existing output tree without rewriting it')
    args = ap.parse_args(argv)

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    if not src.is_dir():
        raise SystemExit(f'source not found: {src}')
    if str(out).startswith(str(src)):
        raise SystemExit('destination must not be inside the source')

    cases = args.cases or sorted((p.name for p in src.iterdir() if p.is_dir()),
                                 key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))

    # One anonymizer for the whole run: it remaps UIDs and shifts dates
    # consistently, so series and study relationships survive intact.
    anonymizer = Anonymizer()
    stats = Counter()
    failures = []

    for case in cases:
        for series in args.series:
            sdir = src / case / series
            if not sdir.is_dir():
                stats['series missing'] += 1
                continue
            files = sorted(sdir.glob('*.dcm'))
            for f in files:
                dest = out / case / series / f.name
                if args.verify_only:
                    frags = identifier_fragments(pydicom.dcmread(str(f), force=True))
                else:
                    frags = deidentify_file(f, dest, anonymizer, stats)
                    
                problems = verify_file(dest, frags, stats)
                if problems:
                    failures.append((dest, problems))
                stats['files'] += 1
        print(f'  case {case}: {stats["files"]} files written', flush=True)

    print(f'\n  files written    : {stats["files"]}')
    print(f'  rescale promoted : {stats["rescale promoted"]}')
    print(f'  series missing   : {stats["series missing"]}')
    print(f'  verification     : {"PASS" if not failures else f"FAILED on {len(failures)} file(s)"}')
    for dest, problems in failures[:10]:
        print(f'    {dest}: {"; ".join(problems)}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
