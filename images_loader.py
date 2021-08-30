from os import walk
import numpy
from PIL import Image

class ImagesLoader:
    def __init__(self, folders_path,type, height=480, width=640, channel_first=False, file_mask=None, postprocessing=None):
        self.channels = 3
        self.height = height
        self.width = width
        self.type = type

        self.postprocessing = postprocessing

        self.file_mask = file_mask

        self.file_names = []
        for folder in folders_path:
            self.file_names = self.file_names + self._find_files(folder)

        self.file_names.sort()

        self.count = len(self.file_names)

        self.channel_first = channel_first

        if self.channel_first:
            self.images = numpy.zeros((self.count, self.channels, self.height, self.width), dtype=numpy.uint8)
        else:
            self.images = numpy.zeros((self.count, self.height, self.width, self.channels), dtype=numpy.uint8)

        ptr = 0
        for file_name in self.file_names:
            print("loading image :", file_name)
            self.images[ptr] = self._load_image(file_name)
            ptr += 1

    def _find_files(self, path):
        #Find files
        files = []
        for (dirpath, dirnames, filenames) in walk(path):
            files.append(filenames)

        result = []
        for file_name in files[0]:
            if file_name.endswith(".jpg") or file_name.endswith(".png"):

                if self.file_mask == None:
                    result.append(path + file_name)
                elif file_name.find(self.file_mask) != -1:
                    result.append(path + file_name)

        return result
    
    def _load_image(self, file_name):
        #Open image
        image = Image.open(file_name).convert("RGB")
        #Mask need to be cropped and images need to be resized
        if self.type == "mask":
            image = image.crop((1, 1, 699, 479))
        else:
            image = image.resize((698, 478))
        
        #Postprocesing for masks, else for images
        if self.postprocessing is not None:
            #Apply postprocesing function on mask
            image = self.postprocessing(image)
            #Mask to numpy array
            image_np = numpy.array(image)
        else:
            #Resize image
            image = image.resize((self.width, self.height))
            #Image to numpy array
            image_np = numpy.array(image)
            #Switch channel
            if self.channel_first and len(image_np.shape) > 2:
                image_np = numpy.moveaxis(image_np, 2, 0)

        return image_np

