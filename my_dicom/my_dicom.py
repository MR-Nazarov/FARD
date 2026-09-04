import numpy as np
import pydicom
import os
from pydicom.uid import generate_uid
import shutil
from copy import deepcopy
import re
import datetime as dt
import traceback
# from dicom.seg2curve import get_curve
import inspect
from tqdm import tqdm
from pathlib2 import Path


class myDicom:

    def __init__(self):

        self.convertFromPacs = False

    def plane(self, dsList):

        planes = []

        if not isinstance(dsList, list):
            dsList = [dsList]

        for ds in dsList:

            if not [0x20, 0x37] in ds:
                return None

            IOP = ds.ImageOrientationPatient
            IOP_round = [round(x) for x in IOP]
            plane = np.cross(IOP_round[0:3], IOP_round[3:6])
            plane = [abs(x) for x in plane]

            if plane[0] == 1:
                planes.append("Sagittal")
            elif plane[1] == 1:
                planes.append("Coronal")
            elif plane[2] == 1:
                planes.append("Axial")

        if all([x == planes[0] for x in planes]):
            return planes[0]
        else:
            return 'undefined'

    def writeDicomSeriesWithObjectAsSource(self, I, sourceObject, output_path, description, correctAirVal=False):

        if (not (os.path.exists(output_path))):
            os.mkdir(output_path)

        print('writing files to ' + output_path)

        RGB = True if len(I.shape) == 4 and I.shape[3] == 3 else False
        numRows = I.shape[1]
        numCols = I.shape[2]

        for i, (dataset, file) in enumerate(sourceObject):

            if (correctAirVal):
                airVal = dataset['0x28', '0x1052'].value

            slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python

            if (slice_number < I.shape[0]):
                if (dataset.pixel_array.dtype == 'uint16'):
                    if (RGB):
                        imToWrite = I[slice_number, :, :, :].astype(np.int8)
                        if (correctAirVal):
                            imToWrite -= np.int8(airVal)
                    else:
                        imToWrite = I[slice_number, :, :].astype(np.int16)
                        if (correctAirVal):
                            imToWrite -= np.int16(airVal)
                    imToWrite[imToWrite < 0] = 0
                    imToWrite = imToWrite.tobytes()
                else:
                    if (RGB):
                        dataset.add_new('0x280006', 'US', 0)
                        imToWrite = I[slice_number, :, :, :].astype(np.int8)
                        if (correctAirVal):
                            imToWrite -= np.int8(airVal)
                        imToWrite = imToWrite.tobytes()
                    else:
                        imToWrite = I[slice_number, :, :].astype(np.int16)
                        if (correctAirVal):
                            imToWrite -= np.int16(airVal)
                        imToWrite = imToWrite.tobytes()
            else:
                continue
            # check if new image has a different shape
            if numRows != dataset[0x28, 0x10].value or numCols != dataset[0x28, 0x11].value:
                elem = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, numRows)
                dataset[0x28, 0x10] = elem
                elem = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, numCols)
                dataset[0x28, 0x11] = elem
            # check rgb
            if RGB:
                elem = pydicom.DataElement((0x28, 0x2), dataset[0x28, 0x2].VR, 3)
                dataset[0x28, 0x2] = elem
                elem = pydicom.DataElement((0x28, 0x4), dataset[0x28, 0x4].VR, 'RGB')
                dataset[0x28, 0x4] = elem
                elem = pydicom.DataElement((0x28, 0x100), dataset[0x28, 0x100].VR, 8)
                dataset[0x28, 0x100] = elem
            # write new image
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, imToWrite)
            dataset[0x7fe0, 0x10] = elem
            # write description
            dataset.SeriesDescription = description
            if (i == 0):
                dicom_uid = generate_uid()
                series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            dataset.SeriesNumber = series_number
            dataset.SOPInstanceUID = generate_uid()
            dataset.SeriesInstanceUID = dicom_uid
            pydicom.dcmwrite(os.path.join(output_path, file), dataset)

    def anonymize(self, dataset, new_person_name="anonymous", new_patient_id="id", remove_curves=True,
                  remove_private_tags=True, remove_technical=True, remove_UID=False):
        """Replace data element values to partly anonymize a DICOM file.
        Note: completely anonymizing a DICOM file is very complicated; there
        are many things this example code does not address. USE AT YOUR OWN RISK.
        """

        # Define call-back functions for the dataset.walk() function
        def UID_callback(ds, data_element):
            if 'uid' in data_element.name.lower():
                del ds[data_element.tag]
            if data_element.tag == (0x40, 0x275):
                del ds[data_element.tag]
            if data_element.tag == [0x8, 0x80]:
                data_element.value = ''

        def PN_callback(ds, data_element):
            """Called from the dataset "walk" recursive function for all data elements."""
            if data_element.VR == "PN":
                data_element.value = new_person_name

        def curves_callback(ds, data_element):
            """Called from the dataset "walk" recursive function for all data elements."""
            if data_element.tag.group & 0xFF00 == 0x5000:
                del ds[data_element.tag]

        def technical_callback(ds, data_element):
            """Called from the dataset "walk" recursive function for all data elements."""
            if data_element.tag.group & 0x00FF == 0x0018:
                data_element.value = ''
            if data_element.tag == [0x8, 0x50]:
                data_element.value = ''

            # Remove patient name and any other person names

        dataset.walk(PN_callback)

        # Change ID
        dataset.PatientID = new_patient_id

        # Remove data elements (should only do so if DICOM type 3 optional)
        # Use general loop so easy to add more later
        # Could also have done: del ds.OtherPatientIDs, etc.
        for name in ['OtherPatientIDs', 'OtherPatientIDsSequence']:
            if name in dataset:
                delattr(dataset, name)

            # Same as above but for blanking data elements that are type 2.
        for name in ['PatientBirthDate', 'PatientAge', 'PatientSex', 'InstitutionAddress', 'InstitutionName']:
            if name in dataset:
                dataset.data_element(name).value = ''

            # Remove private tags if function argument says to do so. Same for curves
        if remove_private_tags:
            dataset.remove_private_tags()
        if remove_curves:
            dataset.walk(curves_callback)
        if remove_technical:
            dataset.walk(technical_callback)
        if remove_UID:
            dataset.walk(UID_callback)

        return dataset

    def writeDicomSeriesOnlyImages(self, I, source_path, output_path, description):

        os.makedirs(output_path, exist_ok=True)
        print('writing files to ' + output_path)
        numRows = I.shape[1]
        numCols = I.shape[2]

        for i, file in enumerate(os.listdir(source_path)):
            if (i > 0 and (I.shape[0] == 1)):
                return
            dataset = pydicom.dcmread(os.path.join(source_path, file))
            slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            if (slice_number < I.shape[0]):
                imToWrite = I[slice_number, :, :].astype(np.int16)
                imToWrite = imToWrite.tobytes()
            else:
                continue
            # check if new image has a different shape
            if (numRows != dataset[0x28, 0x10].value or numCols != dataset[0x28, 0x11].value):
                elem = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, numRows)
                dataset[0x28, 0x10] = elem
                elem = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, numCols)
                dataset[0x28, 0x11] = elem
            # slope = pydicom.DataElement((0x28, 0x1053), dataset[0x28, 0x1053].VR, "1")
            # intercept = pydicom.DataElement((0x28, 0x1052), dataset[0x28, 0x1052].VR, "0")
            # dataset[0x28,0x1053] = slope
            # dataset[0x28,0x1052] = intercept
            elem = pydicom.DataElement((0x28, 0x100), dataset[0x28, 0x100].VR, 32)
            dataset[0x28, 0x100] = elem
            elem = pydicom.DataElement((0x28, 0x101), dataset[0x28, 0x101].VR, 32)
            dataset[0x28, 0x101] = elem
            elem = pydicom.DataElement((0x28, 0x102), dataset[0x28, 0x102].VR, 31)
            dataset[0x28, 0x102] = elem
            # write new image
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, imToWrite)
            dataset[0x7fe0, 0x10] = elem
            # write description
            dataset.SeriesDescription = description
            if (i == 0):
                dicom_uid = generate_uid()
                series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            dataset.SeriesNumber = series_number
            dataset.SOPInstanceUID = generate_uid()
            dataset.SeriesInstanceUID = dicom_uid
            pydicom.dcmwrite(os.path.join(output_path, file), dataset)

    def writeDicomSeriesUsingMeta(self, I,
                                  sourceMeta,
                                  output_path,
                                  description,
                                  fileNames=None,
                                  correctAirVal=False,
                                  sliceNums=None,
                                  calc_suv=True,
                                  metaParams=None,
                                  dtype='int16',
                                  metaValToChange=None):

        os.makedirs(output_path, exist_ok=True)

        print('writing files to ' + output_path)

        RGB = True if len(I.shape) == 4 and I.shape[3] == 3 else False
        numRows = I.shape[1]
        numCols = I.shape[2]

        if fileNames is None:
            fileNames = ['{}.dcm'.format(str(i)) for i in range(len(sourceMeta))]

        for i, dataset in enumerate(sourceMeta):

            file = fileNames[i]

            if correctAirVal:
                airVal = dataset['0x28', '0x1052'].value

            if sliceNums is None:
                if not ['20', '13'] in dataset:
                    continue
                slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            else:
                if i > len(sliceNums) - 1:
                    continue
                slice_number = sliceNums[i]
                elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                dataset[0x20, 0x13] = elem

            if metaParams is not None:
                for Tags in metaParams:
                    tags = Tags.split(',')
                    elem = pydicom.DataElement((tags[0], tags[1]), dataset[tags[0], tags[1]].VR, metaParams[Tags])
                    dataset[tags[0], tags[1]] = elem

            if slice_number >= I.shape[0]:
                slice_number = i
                elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                dataset[0x20, 0x13] = elem

            if dataset.Modality == 'PT' and calc_suv:
                I[slice_number, ...] = self.convert_SUV_to_pixels(I[slice_number, ...], dataset)

            if dataset.pixel_array.dtype == 'uint16':
                if RGB:
                    imToWrite = I[slice_number, :, :, :].astype(np.int8)
                    if correctAirVal:
                        imToWrite -= np.int8(airVal)
                else:
                    if I.dtype != 'uint16':
                        imToWrite = I[slice_number, :, :].astype(dtype)
                    else:
                        imToWrite = I[slice_number, :, :]
                    if correctAirVal:
                        imToWrite -= np.int16(airVal)
                imToWrite[imToWrite < 0] = 0
                imToWrite = imToWrite.tobytes()
            else:
                if RGB:
                    dataset.add_new('0x280006', 'US', 0)
                    imToWrite = I[slice_number, :, :, :].astype(np.int8)
                    if correctAirVal:
                        imToWrite -= np.int8(airVal)
                    imToWrite = imToWrite.tobytes()
                else:
                    imToWrite = I[slice_number, :, :].astype(np.int16)
                    if correctAirVal:
                        imToWrite -= np.int16(airVal)
                    imToWrite = imToWrite.tobytes()
            # check if new image has a different shape
            if numRows != dataset[0x28, 0x10].value or numCols != dataset[0x28, 0x11].value:
                elem = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, numRows)
                dataset[0x28, 0x10] = elem
                elem = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, numCols)
                dataset[0x28, 0x11] = elem
            # check rgb
            if RGB:
                elem = pydicom.DataElement((0x28, 0x2), dataset[0x28, 0x2].VR, 3)
                dataset[0x28, 0x2] = elem
                elem = pydicom.DataElement((0x28, 0x4), dataset[0x28, 0x4].VR, 'RGB')
                dataset[0x28, 0x4] = elem
                elem = pydicom.DataElement((0x28, 0x100), dataset[0x28, 0x100].VR, 8)
                dataset[0x28, 0x100] = elem
            # write new image
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, imToWrite)
            dataset[0x7fe0, 0x10] = elem

            if metaValToChange is not None:

                if isinstance(metaValToChange, list):
                    for M in metaValToChange:
                        elem = pydicom.DataElement((M['tagID'][0], M['tagID'][1]),
                                                   dataset[M['tagID'][0], M['tagID'][1]].VR, M['value'])
                        dataset[M['tagID'][0], M['tagID'][1]] = elem
                else:
                    elem = pydicom.DataElement((metaValToChange['tagID'][0], metaValToChange['tagID'][1]),
                                               dataset[metaValToChange['tagID'][0], metaValToChange['tagID'][1]].VR,
                                               metaValToChange['value'])
                    dataset[metaValToChange['tagID'][0], metaValToChange['tagID'][1]] = elem

            # write description
            if isinstance(description, list) and len(description) > i:
                dataset.SeriesDescription = description[i]
            else:
                dataset.SeriesDescription = description
            if i == 0:
                dicom_uid = generate_uid()
                series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            dataset.SeriesNumber = series_number
            dataset.SOPInstanceUID = generate_uid()
            dataset.SeriesInstanceUID = dicom_uid
            pydicom.dcmwrite(os.path.join(output_path, file), dataset)

    def fix_rep_around(self, im, dtype):

        if (isinstance(dtype, str) and 'float' in dtype) or \
                (isinstance(dtype, np.dtype) and 'float' in dtype.name):
            m = np.finfo(dtype).max
        else:
            m = np.iinfo(dtype).max
        im[im > m] = m

        return im

    def find_max_dtype(self, dtype):

        if (isinstance(dtype, str) and 'float' in dtype) or (isinstance(dtype, np.dtype) and 'float' in dtype.name):
            m = np.finfo(dtype).max
        else:
            m = np.iinfo(dtype).max

        return m

    def writeDicomSeries(self, I,
                         source_path,
                         output_path,
                         description,
                         correctAirVal=False,
                         sliceNums=None,
                         metaParams=None,
                         dtype='int16',
                         metaValToChange=None,
                         locSource=None,
                         anonymize=False,
                         calc_suv=True,
                         onlyReturnDS=False,
                         seg=None,
                         **kwargs):

        if seg is not None:
            seg[seg < 0] = 0
            seg[seg > 0] = 1
            coord, coord_len = get_curve(seg.astype(np.uint8))

        normalizeToBits = kwargs.get('normalizeToBits', None)
        if onlyReturnDS:
            dss = []
        keepMeta = kwargs.get('keepMeta', False)
        zPositions = kwargs.get('zPositions', None)

        # if 'dcm' not in source_path:
        #     self.convertFromPACS(source_path)
        I = self.fix_rep_around(I, dtype=dtype)

        if isinstance(description, list) and len(description) == 1:
            description = description[0]

        if not onlyReturnDS:
            os.makedirs(output_path, exist_ok=True)

        if not kwargs.get('dontPrint', False):
            print('writing files to ' + output_path)

        sameSliceNumber = self.check_duplicate_instance_numbers(source_path)

        RGB = True if len(I.shape) == 4 and I.shape[3] == 3 else False
        if I.ndim == 2:
            I = I[None, :, :]
        numRows = I.shape[-2 - int(RGB)]
        numCols = I.shape[-1 - int(RGB)]

        sourceFiles = [os.path.basename(source_path)] if '.dcm' in source_path else os.listdir(source_path)
        if '.dcm' in source_path:
            source_path = os.path.split(source_path)[0]

        if locSource is not None:
            sourceFilesLoc = os.listdir(locSource)
            sourceFiles = [sourceFiles[0] for i in range(I.shape[0])]

        sliceNumbersWritten = []

        for i, file in enumerate(sourceFiles):

            if i > 0 and (I.shape[0] == 1):
                return

            dataset = pydicom.dcmread(os.path.join(source_path, file), force=True)
            if not hasattr(dataset, 'InstanceNumber'):
                continue

            if sliceNums is None and dataset[0x8, 0x60].value == 'MG':
                sliceNums = np.arange(I.shape[0])

            if anonymize:
                inputArgs = inspect.getfullargspec(self.anonymize).args
                args = {k: kwargs[k] for k in kwargs if k in inputArgs}
                dataset = self.anonymize(dataset, **args)

            if locSource is not None:
                locDS = pydicom.dcmread(os.path.join(locSource, sourceFilesLoc[i]))

            if correctAirVal:
                if ['0x28', '0x1052'] in dataset:
                    airVal = dataset['0x28', '0x1052'].value
                else:
                    correctAirVal = False

            if sliceNums is None:
                if sameSliceNumber:
                    if ['0x7A1', '0x103E'] in dataset:
                        slice_number = int(dataset['0x7A1', '0x103E'].value) - 1
                    else:
                        slice_number = i

                    elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                    dataset[0x20, 0x13] = elem
                else:
                    if not ['20', '13'] in dataset:
                        continue

                    if locSource is not None:
                        slice_number = int(locDS.InstanceNumber) - 1
                        elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                        dataset[0x20, 0x13] = elem
                    else:
                        slice_number = int(
                            dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python

            else:

                if i > len(sliceNums) - 1:
                    continue

                slice_number = sliceNums[i]
                elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                dataset[0x20, 0x13] = elem

            if metaParams is not None:

                for Tags in metaParams:
                    tags = Tags.split(',')
                    elem = pydicom.DataElement((tags[0], tags[1]), dataset[tags[0], tags[1]].VR, metaParams[Tags])
                    dataset[tags[0], tags[1]] = elem

            if slice_number < I.shape[0]:

                if dataset.Modality == 'PT' and calc_suv:
                    I[slice_number, ...] = self.convert_SUV_to_pixels(I[slice_number, ...], dataset)

                if seg is not None:
                    dataset = self.addCurveData(dataset, coord[slice_number], coord_len[slice_number])

                if dataset.pixel_array.dtype == 'uint16':
                    if RGB:
                        imToWrite = I[slice_number, :, :, :].astype('uint8')
                        if correctAirVal:
                            imToWrite -= np.uint8(airVal)
                    else:
                        if I.dtype != 'uint16':
                            imToWrite = I[slice_number, :, :].astype(dtype)
                        else:
                            imToWrite = I[slice_number, :, :]
                        if correctAirVal:
                            imToWrite -= np.int16(airVal)

                    imToWrite[imToWrite < 0] = 0
                    numBits = dataset[0x28, 0x101].value
                    maxVal = 2 ** int(numBits) - 1
                    imToWrite[imToWrite > maxVal] = maxVal
                    imToWrite = imToWrite.tobytes()
                else:
                    if RGB:
                        dataset.add_new('0x280006', 'US', 0)
                        imToWrite = I[slice_number, :, :, :].astype('uint8')
                        if (correctAirVal):
                            imToWrite -= np.uint8(airVal)
                        imToWrite = imToWrite.tobytes()
                    else:

                        imToWrite = I[slice_number, :, :].astype(np.int16)

                        if (correctAirVal):
                            imToWrite -= np.int16(airVal)

                        imToWrite = imToWrite.tobytes()
            else:
                continue

            # check if new image has a different shape
            if numRows != dataset[0x28, 0x10].value or numCols != dataset[0x28, 0x11].value:

                if not kwargs.get('x1y1', None) is None:
                    position = dataset[0x20, 0x32].value
                    x1y1 = kwargs.get('x1y1')
                    spacing = dataset[0x28, 0x30].value  # row spacing \ column spacing
                    position[0] += x1y1[0] * spacing[1]
                    position[1] += x1y1[1] * spacing[0]
                    elem = pydicom.DataElement((0x20, 0x32), dataset[0x20, 0x32].VR, position)
                    dataset[0x20, 0x32] = elem

                elem = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, numRows)
                dataset[0x28, 0x10] = elem
                elem = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, numCols)
                dataset[0x28, 0x11] = elem
            # check rgb
            if RGB:
                elem = pydicom.DataElement((0x28, 0x2), dataset[0x28, 0x2].VR, 3)
                dataset[0x28, 0x2] = elem
                elem = pydicom.DataElement((0x28, 0x4), dataset[0x28, 0x4].VR, 'RGB')
                dataset[0x28, 0x4] = elem
                elem = pydicom.DataElement((0x28, 0x100), dataset[0x28, 0x100].VR, 8)
                dataset[0x28, 0x100] = elem
            # write new image
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, imToWrite)
            dataset[0x7fe0, 0x10] = elem

            if locSource is not None:
                elem = pydicom.DataElement((0x20, 0x1041), dataset[0x20, 0x1041].VR, locDS[0x20, 0x1041].value)
                dataset[0x20, 0x1041] = elem
                elem = pydicom.DataElement((0x20, 0x32), dataset[0x20, 0x32].VR, locDS[0x20, 0x32].value)
                dataset[0x20, 0x32] = elem
                elem = pydicom.DataElement((0x18, 0x50), dataset[0x18, 0x50].VR, locDS[0x18, 0x50].value)
                dataset[0x18, 0x50] = elem
                elem = pydicom.DataElement((0x28, 0x30), dataset[0x28, 0x30].VR, locDS[0x28, 0x30].value)
                dataset[0x28, 0x30] = elem

            if zPositions is not None:
                currentPositions = dataset[(0x20, 0x32)].value
                newPostions = currentPositions
                newPostions[-1] = zPositions[i]
                elem = pydicom.DataElement((0x20, 0x32), dataset[0x20, 0x32].VR, newPostions)
                dataset[0x20, 0x32] = elem

            if metaValToChange is not None:

                if isinstance(metaValToChange, list):
                    for M in metaValToChange:
                        elem = pydicom.DataElement((M['tagID'][0],
                                                    M['tagID'][1]),
                                                   dataset[M['tagID'][0],
                                                           M['tagID'][1]].VR,
                                                   M['value'])
                        dataset[M['tagID'][0], M['tagID'][1]] = elem
                else:
                    elem = pydicom.DataElement((metaValToChange['tagID'][0],
                                                metaValToChange['tagID'][1]),
                                               dataset[metaValToChange['tagID'][0],
                                                       metaValToChange['tagID'][1]].VR,
                                               metaValToChange['value'])
                    dataset[metaValToChange['tagID'][0], metaValToChange['tagID'][1]] = elem

            # write description
            if not keepMeta:
                if isinstance(description, list) and len(description) > i:
                    if description[i] != '':
                        if '+' in description:
                            description = exec(description.replace('description', dataset.SeriesDescription))
                        dataset.SeriesDescription = description[i]
                elif description != '':
                    if '+' in description:
                        description = description.split('+')
                        description = [x.replace('description', dataset.SeriesDescription) for x in description]
                        description = ''.join(description)
                    elem = pydicom.DataElement((0x8, 0x103E), dataset[0x8, 0x103E].VR, description)
                    dataset[0x8, 0x103E] = elem

                if i == 0:
                    dicom_uid = generate_uid()
                    series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))

                dataset.SeriesNumber = series_number
                dataset.SOPInstanceUID = generate_uid()
                dataset.SeriesInstanceUID = dicom_uid

            if slice_number in sliceNumbersWritten:
                continue

            if not onlyReturnDS:
                if locSource is not None:
                    pydicom.dcmwrite(os.path.join(output_path, sourceFilesLoc[i]), dataset)
                else:
                    pydicom.dcmwrite(os.path.join(output_path, file), dataset)
            else:
                dss.append(dataset)

            sliceNumbersWritten.append(slice_number)
        if onlyReturnDS:
            return dss

    def read_as_reversed(self, dicomDir):
        I, meta = self.readDicomSeriesWithMeta(dicomDir)
        J = np.zeros(I.shape, dtype=I.dtype)
        numSlices = I.shape[0]
        for idx in range(I.shape[0]):
            J[idx, :, :] = I[numSlices - idx - 1, :, :]
        return I[::-1, :, :], meta

    def writeDicom(self, path, source_path, I, description, dtype='int16', anonymize=False):

        if I.ndim > 2:
            self.writeDicomSeries(I=I, source_path=os.path.dirname(source_path), output_path=os.path.dirname(path),
                                  description=description, sliceNums=[0])
        else:
            I = self.fix_rep_around(I, dtype=dtype)
            path_dir = '/'.join(path.split('/')[0:-1])
            os.makedirs(path_dir, exist_ok=True)
            dataset = pydicom.dcmread(source_path)
            if anonymize:
                dataset = self.anonymize(dataset)
            dicom_uid = generate_uid()
            series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, I.astype(np.int16).tobytes())
            dataset[0x7fe0, 0x10] = elem
            dataset.SeriesNumber = series_number
            dataset.SOPInstanceUID = generate_uid()
            dataset.SeriesInstanceUID = dicom_uid
            if (description != ''):
                dataset.SeriesDescription = description
            pydicom.dcmwrite(path, dataset)
            print('writing file to {}'.format(path))

    def readDicomSeries(self, path):

        self.convertFromPACS(path)

        dicomFiles = os.listdir(path)
        numFiles = len(dicomFiles)

        sameSliceNumber = self.check_duplicate_instance_numbers(path)

        for i, file in enumerate(dicomFiles):

            if (not 'dcm' in os.path.join(path, file)):
                continue

            dataset = pydicom.dcmread(os.path.join(path, file))
            if (sameSliceNumber):
                slice_number = int(dataset['0x7A1', '0x103E'].value) - 1
            else:
                slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            if (i == 0):
                RGB = True if dataset[0x28, 0x2].value == 3 else False
                if (RGB):
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1],
                                  dataset.pixel_array.shape[2]),
                                 dtype=dataset.pixel_array.dtype)
                else:
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1]),
                                 dtype=dataset.pixel_array.dtype)
                self.airVal = dataset['0x28', '0x1052'].value
            tmpIm = dataset.pixel_array
            tmpIm[tmpIm < self.airVal] = 0
            I[slice_number, :, :] = tmpIm

        return I

    def readDicomSeriesWithMetaAndFileName(self, path):

        self.convertFromPACS(path)

        dicomFiles = os.listdir(path)
        numFiles = len(dicomFiles)
        meta = [None] * numFiles

        sameSliceNumber = self.check_duplicate_instance_numbers(path)

        for i, file in enumerate(dicomFiles):

            if (not 'dcm' in os.path.join(path, file)):
                continue

            dataset = pydicom.dcmread(os.path.join(path, file))
            if (sameSliceNumber):
                slice_number = int(dataset['0x7A1', '0x103E'].value) - 1
            else:
                slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            if (i == 0):
                RGB = True if dataset[0x28, 0x2].value == 3 else False
                if (RGB):
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1],
                                  dataset.pixel_array.shape[2]),
                                 dtype=dataset.pixel_array.dtype)
                else:
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1]),
                                 dtype=dataset.pixel_array.dtype)
                self.airVal = dataset['0x28', '0x1052'].value
            if (slice_number < len(meta)):
                meta[slice_number] = [dataset, file]
            tmpIm = dataset.pixel_array
            tmpIm[tmpIm < self.airVal] = 0
            if (slice_number < I.shape[0]):
                if (RGB):
                    I[slice_number, :, :, :] = tmpIm
                else:
                    I[slice_number, :, :] = tmpIm

        return I, meta

    def readDicomSeriesWithMeta(self, path,
                                insertInstanceNumber=False,
                                calc_suv=True,
                                only_first_dcm=False,
                                dcm_idx=0,
                                useForceRead=False,
                                printProcess=True):

        singleDicom = False
        if not os.path.isdir(path):
            dicomFiles = [path]
            singleDicom = True
        else:
            self.convertFromPACS(path)
            dicomFiles = os.listdir(path)

        self.sliceNumsFilenames = []

        dicomFiles = dicomFiles if not only_first_dcm else dicomFiles[dcm_idx: dcm_idx + 1]
        numFiles = len(dicomFiles)
        meta = [None] * numFiles

        if not singleDicom:
            sameSliceNumber = self.check_duplicate_instance_numbers(path)
        else:
            sameSliceNumber = False

        if printProcess:
            loader = tqdm(dicomFiles, position=0, leave=True)
        else:
            loader = dicomFiles
        I = None

        for i, file in enumerate(loader):

            dataset = pydicom.dcmread(os.path.join(path, file), force=useForceRead)
            if not (0x7FE0, 0x10) in dataset:
                continue

            if not insertInstanceNumber and dataset[0x8, 0x60].value == 'MG':
                insertInstanceNumber = True

            if (sameSliceNumber and not ['0x7A1', '0x103E'] in dataset):
                insertInstanceNumber = True

            if (sameSliceNumber and ['0x7A1', '0x103E'] in dataset):
                slice_number = int(dataset['0x7A1', '0x103E'].value) - 1
            else:
                if (not ['20', '13'] in dataset):
                    continue
                if (not insertInstanceNumber):
                    slice_number = int(
                        dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
                else:
                    slice_number = i

                if only_first_dcm or singleDicom:
                    slice_number = 0

            if (i == 0):
                RGB = True if (0x28, 0x2) in dataset and dataset[0x28, 0x2].value == 3 else False
                if (RGB):
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1],
                                  dataset.pixel_array.shape[2]), dtype=dataset.pixel_array.dtype)
                else:
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1]),
                                 dtype=dataset.pixel_array.dtype)
                if [0x28, 0x1052] in dataset:
                    self.airVal = float(dataset['0x28', '0x1052'].value)
                else:
                    self.airVal = 0
            else:
                if not isinstance(I, list) and I.shape[1:] != dataset.pixel_array.shape:
                    I_ = I
                    I = [None] * I_.shape[0]
                    for j in range(i):
                        I[j] = I_[j, ...]

            if slice_number < len(meta):
                meta[slice_number] = dataset
            tmpIm = dataset.pixel_array
            tmpIm[tmpIm < self.airVal] = 0

            if dataset.Modality == 'PT':

                if calc_suv:
                    tmpIm = tmpIm * float(dataset[0x28, 0x1053].value) + float(dataset[0x28, 0x1052].value)
                    tmpIm = self.calc_suv_func(tmpIm, dataset)
                    I = I.astype('float32')

                currentMaxDtype = self.find_max_dtype(I.dtype)
                if tmpIm.max() > currentMaxDtype:
                    match = re.match(r"([a-z]+)([0-9]+)", str(dataset.pixel_array.dtype), re.I)
                    split = match.groups()
                    newDtype = split[0] + str(int(split[1]) * 2)
                    I = I.astype(newDtype)

            _slice = None
            if isinstance(I, list):
                I[i] = tmpIm
                _slice = i
            else:
                if (slice_number < I.shape[0]):
                    I[slice_number, ...] = tmpIm
                    _slice = slice_number

            if not _slice is None:
                self.sliceNumsFilenames.append({
                    'slice': _slice,
                    'filename': Path(path, file)})

        instances = [x for x in range(len(meta)) if not meta[x] is None]

        if len(instances) != numFiles:
            meta = [meta[x] for x in instances]
            I = I[instances]

        return I, meta

    def calc_suv_func(self, Ac, ds):

        startTime = ds[0x8, 0x32].value
        startTime = dt.datetime.strptime(startTime, '%H%M%S')
        weight = float(ds[0x10, 0x1030].value) * 1e3
        doseInfo = ds[0x54, 0x16][0]
        totalDose = float(doseInfo[0x18, 0x1074].value)
        injectionTime = doseInfo[0x18, 0x1072].value
        injectionTime = dt.datetime.strptime(injectionTime, '%H%M%S')
        halfLife = float(doseInfo[0x18, 0x1075].value)
        deltaT = (startTime - injectionTime).seconds

        num = weight * Ac
        denom = totalDose * (2 ** (-deltaT / halfLife))
        SUV = num / denom

        return SUV

    def convert_SUV_to_pixels(self, suv, ds):

        startTime = ds[0x8, 0x32].value
        startTime = dt.datetime.strptime(startTime, '%H%M%S')
        weight = float(ds[0x10, 0x1030].value) * 1e3
        doseInfo = ds[0x54, 0x16][0]
        totalDose = float(doseInfo[0x18, 0x1074].value)
        injectionTime = doseInfo[0x18, 0x1072].value
        injectionTime = dt.datetime.strptime(injectionTime, '%H%M%S')
        halfLife = float(doseInfo[0x18, 0x1075].value)
        deltaT = (startTime - injectionTime).seconds

        denom = totalDose * (2 ** (-deltaT / halfLife))
        Ac = (denom * suv) / weight
        Ac = (Ac - float(ds[0x28, 0x1052].value)) / float(ds[0x28, 0x1053].value)

        Ac = self.fix_rep_around(Ac[None, ...], dtype='int16')[0, ...]
        Ac[Ac < 0] = 0

        return Ac

    def check_duplicate_instance_numbers(self, path):

        if 'dcm' in path:
            return False

        dicomFiles = os.listdir(path)
        hasDuplicates = False
        sliceNumbers = []

        for i, file in enumerate(dicomFiles):

            dataset = pydicom.dcmread(os.path.join(path, file), force=True)

            if not hasattr(dataset, 'InstanceNumber'):
                continue

            sliceNumber = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            sliceNumbers.append(sliceNumber)
            slices = [x for x in sliceNumbers if x == sliceNumber]
            if (len(slices) > 2):
                hasDuplicates = True
                break
            if (i > 3):
                break

        return hasDuplicates

    def readDicom(self, path):
        dataset = pydicom.dcmread(path)
        tmpIm = dataset.pixel_array
        if hasattr(dataset, 'RescaleIntercept'):
            self.airVal = dataset['0x28', '0x1052'].value
            tmpIm[tmpIm < self.airVal] = 0
        return tmpIm

    def readDicomWithMeta(self, path):
        dataset = pydicom.dcmread(path)
        tmpIm = dataset.pixel_array
        if hasattr(dataset, 'RescaleIntercept'):
            self.airVal = dataset['0x28', '0x1052'].value
            tmpIm[tmpIm < self.airVal] = 0
        return tmpIm, dataset

    def convertFromPACS(self, directory):
        files = os.listdir(directory)
        if len(files) == 0:
            return
        dicom_files = [f for f in files if 'dcm' in f]
        if len(dicom_files) == 0 and self.convertFromPacs:
            print('converting the following directory: ' + directory)
            [os.remove(os.path.join(directory, f)) for f in files if
             (not os.path.isdir(os.path.join(directory, f)) and '.' not in f)]
            for folder, subs, files in os.walk(directory):
                if 'VERSION' in files and len(subs) == 0:
                    os.remove(os.path.join(folder, 'VERSION'))
                    for f in files:
                        if 'VERSION' in f:
                            continue
                        new_file = os.path.join(folder, f + '.dcm')
                        os.rename(os.path.join(folder, f), new_file)
                        shutil.copy2(new_file, directory)
                    break
            files = os.listdir(directory)
            [shutil.rmtree(os.path.join(directory, f))
             for f in files if os.path.isdir(os.path.join(directory, f))]
        elif len(dicom_files) == 0:
            print('no dicom files in {}, maybe convert from PACS? '
                  'add convertFromPacs = True for conversion'.format(directory))

    def convertFolderFromPACS(self, folder):
        dirs = os.listdir(folder)
        for d in dirs:
            self.convertFromPACS(os.path.join(folder, d))
