import numpy as np
import pydicom
import os
from pydicom.uid import generate_uid
import shutil
import cv2


class myDicom:

    def __init__(self, convertFromPacs=False):
        self.convertFromPacs = convertFromPacs

    def checkSquareImage(self, path):

        if('dcm' in path):
            I, dataset = self.readDicomWithMeta(path)
            Rows = I.shape[0]
            Columns = I.shape[1]
            if Rows != Columns:
                length = min(Rows, Columns)
                I_crop = I[(Rows - length) / 2:length + (Rows - length) / 2,
                         (Columns - length) / 2:length + (Columns - length) / 2]

                self.writeDicom( I_crop, path, path)
        else:
            I, dataset = self.readDicomSeriesWithMeta(path)
            Rows = I[0].shape[0]
            Columns = I[0].shape[1]
            if Rows!=Columns:
                length=min(Rows,Columns)
                I_crop=I[:,int((Rows-length)/2):int(length+(Rows-length)/2),int((Columns-length)/2):int(length+(Columns-length)/2) ]
                self.writeDicomSeries( I_crop, path, path)

    def writeDicomSeries(self,I,source_path,output_path,description='',interp=1):

        if (not (os.path.exists(output_path))):
            os.makedirs(output_path, exist_ok=False)

        sameSliceNumber = self.check_duplicate_instance_numbers(source_path)

        for i,file in enumerate(os.listdir(source_path)):
            if(i > 0 and (I.shape[0] == 1)):
                return
            dataset = pydicom.dcmread(os.path.join(source_path,file))

            if (sameSliceNumber):
                slice_number = int(dataset['0x7A1', '0x103E'].value) - 1
                elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                dataset[0x20, 0x13] = elem
            else:
                slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python

            if(self.nonSeries):
                slice_number = i

            if(slice_number < I.shape[0]):
                if (dataset.pixel_array.dtype == 'uint16'):
                    imToWrite = I[slice_number, :, :].astype(np.int16)
                    imToWrite[imToWrite < 0] = 0
                    imToWrite = imToWrite.tobytes()
                else:
                    imToWrite = I[slice_number, :, :].astype(np.int16).tobytes()
            else:
                continue
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0,0x10].VR ,imToWrite)
            Rows = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, I.shape[1])
            Columns = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, I.shape[2])
            dataset[0x28, 0x10] = Rows
            dataset[0x28, 0x11] = Columns
            dataset[0x7fe0, 0x10] = elem
            if (description != ''):
                dataset.SeriesDescription = description



            if (interp != 1):
                # prev_Pxl_spacing = list(map(int,dataset['0x28', '0x30'].value[0]))
                prev_Pxl_spacing=[dataset['0x28', '0x30'].value[0], dataset['0x28', '0x30'].value[1]]
                # prev_Pxl_spacing = list(map(float,prev_Pxl_spacing))
                Pxl_spacing_val = list(map(lambda x: (1/interp) * x, prev_Pxl_spacing))

                Pxl_spacing = pydicom.DataElement((0x28, 0x30), dataset[0x28, 0x30].VR,Pxl_spacing_val)
                dataset[0x28, 0x30]=Pxl_spacing

            if (i == 0):
                dicom_uid = generate_uid()
                series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            dataset.SeriesNumber = series_number
            dataset.SOPInstanceUID = generate_uid()
            dataset.SeriesInstanceUID = dicom_uid
            print('writing file ' + os.path.join(output_path,file))
            pydicom.dcmwrite(os.path.join(output_path,file), dataset)

    def change_meta_from_axial_to_sagittal(self,I, meta, output_path,description=''):

        if (not (os.path.exists(output_path))):
            os.makedirs(output_path, exist_ok=True)


        if len(meta)<I.shape[0]: #we had less slices in axial rather than sagittal view, so lets duplicate and re-write info later
            for i in range(int(I.shape[0]-len(meta))):
                meta.append(meta[-1])

        for sliceNum in range(I.shape[0]):
            meta[sliceNum].Rows = I.shape[1]
            meta[sliceNum].Columns = I.shape[2]
            meta[sliceNum].ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
            if (sliceNum == 0):
                dicom_uid = generate_uid()
                series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            meta[sliceNum].SeriesNumber = series_number
            meta[sliceNum].SOPInstanceUID = generate_uid()
            # meta[sliceNum].MediaStorageSOPInstanceUID = meta[sliceNum].SOPInstanceUID
            meta[sliceNum].SeriesInstanceUID = dicom_uid
            meta[sliceNum].InstanceNumber = sliceNum+1
            # meta[sliceNum].ImagePositionPatient = [] #todo:add

            if (sliceNum < I.shape[0]):
                if (meta[sliceNum].pixel_array.dtype == 'uint16'):
                    imToWrite = I[sliceNum, :, :].astype(np.int16)
                    imToWrite[imToWrite < 0] = 0
                    imToWrite = imToWrite.tobytes()
                else:
                    imToWrite = I[sliceNum, :, :].astype(np.int16).tobytes()
            else:
                continue

            elem = pydicom.DataElement((0x7fe0, 0x10), meta[sliceNum][0x7fe0, 0x10].VR, imToWrite)
            Rows = pydicom.DataElement((0x28, 0x10), meta[sliceNum][0x28, 0x10].VR, I.shape[1])
            Columns = pydicom.DataElement((0x28, 0x11), meta[sliceNum][0x28, 0x11].VR, I.shape[2])
            meta[sliceNum][0x28, 0x10] = Rows
            meta[sliceNum][0x28, 0x11] = Columns
            meta[sliceNum][0x7fe0, 0x10] = elem

            if (description != ''):
                meta[sliceNum].SeriesDescription = description

            print('writing file ' + os.path.join(output_path, str(sliceNum+1)+'.dcm'))
            pydicom.dcmwrite(os.path.join(output_path, str(sliceNum+1))+'.dcm', meta[sliceNum])

        """
        fields to edit 
        ROWS V
        COLS V
        iNSTANCE NUMBER V
        Image Orientation (Patient) for sagittal write this: ['0', '1', '0', '0', '0', '-1']  V
        ImagePositionPatient this also needs to be changed 
        SliceThickness
        Pixel Spacing  if Pixel Spacing equals to both values of SliceThickness we don't need to change these values
        also, since we will have more slices in this specific case we also need to add (0002, 0003) Media Storage SOP Instance UID
        using pydicom generate_uid()
        and (0008, 0018) SOP Instance UID   
        """

    def writeDicom(self,I,source_path,output_path,description='',interp=1):
        path_dir = '/'.join(output_path.split('/')[0:-1])
        if (not (os.path.exists(path_dir))):
            os.mkdir(path_dir)
        dataset = pydicom.dcmread(source_path)
        elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, I.astype(np.int16).tobytes())
        Rows = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, I.shape[0])
        Columns = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, I.shape[1])
        dataset[0x28, 0x10] = Rows
        dataset[0x28, 0x11] = Columns
        dataset[0x7fe0, 0x10] = elem
        if (interp != 1):
            # prev_Pxl_spacing = list(map(int,dataset['0x28', '0x30'].value[0]))
            prev_Pxl_spacing = [dataset['0x28', '0x30'].value[0], dataset['0x28', '0x30'].value[1]]
            # prev_Pxl_spacing = list(map(float,prev_Pxl_spacing))
            Pxl_spacing_val = list(map(lambda x: (1 / interp) * x, prev_Pxl_spacing))

            Pxl_spacing = pydicom.DataElement((0x28, 0x30), dataset[0x28, 0x30].VR, Pxl_spacing_val)
            dataset[0x28, 0x30] = Pxl_spacing
        if(description != ''):
            dataset.SeriesDescription = description
        pydicom.dcmwrite(output_path, dataset)

    def readDicomSeries(self,path):

        if (self.convertFromPacs):
            self.convertFromPACS(path)

        dicomFiles = os.listdir(path)
        numFiles = len(dicomFiles)

        sameSliceNumber = self.check_duplicate_instance_numbers(path)

        for i,file in enumerate(dicomFiles):

            dataset = pydicom.dcmread(os.path.join(path,file))
            if(sameSliceNumber):
                slice_number = int(dataset['0x7A1','0x103E'].value) - 1
            else:
                slice_number = int(dataset.InstanceNumber) - 1 # InstanceNumber starts from 1 and not zero as in python
            if (self.nonSeries):
                slice_number = i
            if(i == 0):
                I = np.zeros((numFiles,dataset.pixel_array.shape[0],dataset.pixel_array.shape[1]),dtype = dataset.pixel_array.dtype)
                self.airVal = dataset['0x28', '0x1052'].value
            tmpIm = dataset.pixel_array
            tmpIm[tmpIm < self.airVal] = 0
            I[slice_number, :, :] = tmpIm

        return I

    def readDicomSeriesWithMeta(self,path):

        if(self.convertFromPacs):
            self.convertFromPACS(path)

        dicomFiles = os.listdir(path)
        numFiles = len(dicomFiles)
        meta = [None] * numFiles

        self.check_duplicate_instance_numbers(path)

        for i,file in enumerate(dicomFiles):
            # print(os.path.join(path,file))
            dataset = pydicom.dcmread(os.path.join(path,file))#force = true
            # if(sameSliceNumber):
            #     slice_number = int(dataset['0x7A1','0x103E'].value) - 1
            # else:
            slice_number = int(dataset.InstanceNumber) - 1 # InstanceNumber starts from 1 and not zero as in python
            if (self.nonSeries):
                slice_number = i
            if(i == 0):
                I = np.zeros((numFiles,dataset.pixel_array.shape[0],dataset.pixel_array.shape[1]),dtype = np.int16)
                if ['0x28', '0x1052'] in dataset:
                    self.airVal = dataset['0x28', '0x1052'].value
            if(slice_number < len(meta)):
                meta[slice_number] = dataset
            tmpIm = dataset.pixel_array
            if hasattr(self,  'airVal'):
                tmpIm[tmpIm < self.airVal] = 0
            if(slice_number < I.shape[0]):
                I[slice_number, :, :] = tmpIm

        return I,meta

    def check_duplicate_instance_numbers(self,path):

        dicomFiles = os.listdir(path)
        hasDuplicates = False
        sliceNumbers = []
        # self.nonSeries = True
        self.nonSeries = False

        numFiles = len(dicomFiles)

        for i, file in enumerate(dicomFiles):
            dataset = pydicom.dcmread(os.path.join(path, file))
            sliceNumber = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            if(not self.nonSeries and dataset.InstanceNumber > numFiles):
                self.nonSeries = True
            sliceNumbers.append(sliceNumber)
            slices = [x for x in sliceNumbers if x == sliceNumber]
            if(len(slices) > 1):
                # hasDuplicates = True
                self.nonSeries = True
                break
            if(i > 3):
                break

        # return hasDuplicates

    def readDicom(self,path):
        dataset = pydicom.dcmread(path)
        tmpIm = dataset.pixel_array
        # self.airVal = dataset['0x28', '0x1052'].value
        # tmpIm[tmpIm < self.airVal] = 0
        try:
            self.airVal = dataset['0x28', '0x1052'].value
            tmpIm[tmpIm < self.airVal] = 0
        except:
            if tmpIm.min()==0:
                self.airVal=0
                tmpIm[tmpIm < self.airVal] = 0
            else:
                print(tmpIm.min())
                raise ValueError
        return tmpIm

    def readDicomWithMeta(self,path):
        dataset = pydicom.dcmread(path)
        tmpIm = dataset.pixel_array
        # self.airVal = dataset['0x28', '0x1052'].value
        # tmpIm[tmpIm < self.airVal] = 0
        try:
            self.airVal = dataset['0x28', '0x1052'].value
            tmpIm[tmpIm < self.airVal] = 0
        except:
            if tmpIm.min()==0:
                self.airVal=0
                tmpIm[tmpIm < self.airVal] = 0
            else:
                print(tmpIm.min())
                raise ValueError
        return tmpIm, dataset

    def convertFromPACS(self,directory):
        files = os.listdir(directory)
        non_dicom = [f for f in files if (not 'dcm' in f)]
        if (len(non_dicom) > 0):
            [os.remove(os.path.join(directory, f)) for f in files if (not os.path.isdir(os.path.join(directory, f)))]
            for folder, subs, files in os.walk(directory):
                if ('VERSION' in files and len(files) > 4):
                    os.remove(os.path.join(folder, 'VERSION'))
                    for f in files:
                        if ('VERSION' in f):
                            continue
                        new_file = os.path.join(folder, f + '.dcm')
                        os.rename(os.path.join(folder, f), new_file)
                        shutil.copy2(new_file, directory)
                    break
            files = os.listdir(directory)
            [shutil.rmtree(os.path.join(directory, f)) for f in files if os.path.isdir(os.path.join(directory, f))]

    def convertFolderFromPACS(self,folder):
        dirs = os.listdir(folder)
        for d in dirs:
            print('converting the following directory: ' + d)
            self.convertFromPACS(os.path.join(folder,d))


    def convert_save_png_jpg(self, source_path,output_path,PNG=True):
        # source_path: the .dcm folder path
        # output_path: the output jpg/png folder path
        images_path = os.listdir(source_path)
        for n, image in enumerate(images_path):
            ds = pydicom.dcmread(os.path.join(source_path, image))
            pixel_array_numpy = ds.pixel_array
            if PNG == False:
                image = image.replace('.dcm', '.jpg')
            else:
                image = image.replace('.dcm', '.png')
            cv2.imwrite(os.path.join(output_path, image), pixel_array_numpy)
            if n % 50 == 0:
                print('{} image converted'.format(n))


    def anonymize(self, dataset, new_person_name="anonymous", new_patient_id="id", remove_curves=True, remove_private_tags=True, remove_technical = True):
        """Replace data element values to partly anonymize a DICOM file.
        Note: completely anonymizing a DICOM file is very complicated; there
        are many things this example code does not address. USE AT YOUR OWN RISK.
        """

        # Define call-back functions for the dataset.walk() function
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
            if data_element.tag == [0x8,0x80]:
                data_element.value = 'Sheba'
            if data_element.tag == [0x8,0x50]:
                data_element.value = ''
            if data_element.tag == [0x20, 0x10]: #Study ID
                data_element.value = '11111111'
            if data_element.tag == [0x8, 0x1030]: #Study Description
                data_element.value = 'MRI T2'
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

        return dataset

    def writeDicomSeriesUsingMeta(self, I, sourceMeta, output_path, description, fileNames=None, correctAirVal=False,
                                  sliceNums=None, metaParams=None, dtype='int16', metaValToChange=None):

        os.makedirs(output_path, exist_ok=True)

        print('writing files to ' + output_path)

        RGB = True if len(I.shape) == 4 and I.shape[3] == 3 else False
        numRows = I.shape[1]
        numCols = I.shape[2]

        if fileNames is None:
            fileNames = ['{}.dcm'.format(str(i)) for i in range(len(sourceMeta))]

        for i, dataset in enumerate(sourceMeta):

            file = fileNames[i]

            if (correctAirVal):
                airVal = dataset['0x28', '0x1052'].value

            if (sliceNums is None):
                if (not ['20', '13'] in dataset):
                    continue
                slice_number = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            else:
                if (i > len(sliceNums) - 1):
                    continue
                slice_number = sliceNums[i]
                elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                dataset[0x20, 0x13] = elem

            if (not metaParams is None):
                for Tags in metaParams:
                    tags = Tags.split(',')
                    elem = pydicom.DataElement((tags[0], tags[1]), dataset[tags[0], tags[1]].VR, metaParams[Tags])
                    dataset[tags[0], tags[1]] = elem

            if (slice_number >= I.shape[0]):
                slice_number = i
                elem = pydicom.DataElement((0x20, 0x13), dataset[0x20, 0x13].VR, slice_number + 1)
                dataset[0x20, 0x13] = elem
            if (dataset.pixel_array.dtype == 'uint16'):
                if (RGB):
                    imToWrite = I[slice_number, :, :, :].astype(np.int8)
                    if (correctAirVal):
                        imToWrite -= np.int8(airVal)
                else:
                    if (I.dtype != 'uint16'):
                        imToWrite = I[slice_number, :, :].astype(dtype)
                    else:
                        imToWrite = I[slice_number, :, :]
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
            # check if new image has a different shape
            if (numRows != dataset[0x28, 0x10].value or numCols != dataset[0x28, 0x11].value):
                elem = pydicom.DataElement((0x28, 0x10), dataset[0x28, 0x10].VR, numRows)
                dataset[0x28, 0x10] = elem
                elem = pydicom.DataElement((0x28, 0x11), dataset[0x28, 0x11].VR, numCols)
                dataset[0x28, 0x11] = elem
            # check rgb
            if (RGB):
                elem = pydicom.DataElement((0x28, 0x2), dataset[0x28, 0x2].VR, 3)
                dataset[0x28, 0x2] = elem
                elem = pydicom.DataElement((0x28, 0x4), dataset[0x28, 0x4].VR, 'RGB')
                dataset[0x28, 0x4] = elem
                elem = pydicom.DataElement((0x28, 0x100), dataset[0x28, 0x100].VR, 8)
                dataset[0x28, 0x100] = elem
            # write new image
            elem = pydicom.DataElement((0x7fe0, 0x10), dataset[0x7fe0, 0x10].VR, imToWrite)
            dataset[0x7fe0, 0x10] = elem

            if not metaValToChange is None:

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
            if (isinstance(description, list) and len(description) > i):
                dataset.SeriesDescription = description[i]
            else:
                dataset.SeriesDescription = description
            if (i == 0):
                dicom_uid = generate_uid()
                series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
            dataset.SeriesNumber = series_number
            dataset.SOPInstanceUID = generate_uid()
            dataset.SeriesInstanceUID = dicom_uid
            pydicom.dcmwrite(os.path.join(output_path, file), dataset)