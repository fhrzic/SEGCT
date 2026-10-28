""" Config for the inference of the model"""
import json
import os

class InferenceConfig:
    '''
    Class for specifying inference parameters to perfrom it on any data sample
    '''
    '''
    model_path: path to the model weights and whole structure. The complete model is saved
    '''
    model_path = None

    '''
    model_path: path to the model config. All data about the model including patch_size. This can
    be also set to be a tupple i.e.(128,128,256). Only this is important.
    '''
    model_config = None

    '''
    output_dir: dir where the outputs will be created
    '''
    output_dir = None

    '''
    device: device where the model and data will do inference (torch style e.g. cuda:1)
    '''
    device = 'cpu'

    '''
    verbose: verbose to print different useful info
    '''
    verbose = True

    '''
    normalization_fingerprint: path to the normalization fingerprint file. This is
    necessary to know how to preprocess the data. Unless already preprocessed data
    is provided
    '''
    normalization_fingerprint = None

    '''
    data_fingerprint: path to the data fingerprint file. 
    '''
    data_fingerprint = None

    '''
    patch_shape: tuple (x,y,z) of a patch shape to which the input volume should be splitted to. 
    This must be the same size as volume
    '''
    patch_shape = None

    '''
    batch_size, integer, number of patches which can be simultaniously processed on the gpu
    '''
    batch_size = 1

    '''
    overlap, float, this defines for how much will patches overlap. The default value is 0
    which means no overlap. E.G. 0.5 means there is overlap of 50% between patches
    '''
    overlap = 0.5

    '''
    gaussian, float, gaussian function apply on the patches
    '''
    gaussian = False

    '''
    flip, bool, if set to true the patch will be flipped three times (across every axis) when
    predicting the segmentation for that map. Adds time - but hopefully also adds accuracy
    '''
    flip = True

    def __str__(self):
        return (
            "######################################\n"
            "InferenceConfig:\n"
            f"  model_path: {self.model_path}\n"
            f"  model_config: {self.model_config}\n"
            f"  output_dir: {self.output_dir}\n"
            f"  device: {self.device}\n"
            f"  normalization_fingerprint: {self.normalization_fingerprint}\n"
            f"  data_fingerprint: {self.data_fingerprint}\n"
            f"  patch_shape: {self.patch_shape}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  overlap: {self.overlap}\n"
            f"  gaussian: {self.gaussian}\n"
            f"  flip: {self.flip}\n"
            f"  verbose: {self.verbose}\n"
            "######################################"
        )

    def export_to_json(self, export_path: str = None):
        _output_dict = {
            "model_path":                 self.model_path,
            "model_config":               self.model_config,
            "output_dir":                 self.output_dir,
            "device":                     self.device,
            "normalization_fingerprint":  self.normalization_fingerprint,
            "data_fingerprint":           self.data_fingerprint,
            "patch_shape":                self.patch_shape,
            "batch_size":                 self.batch_size,
            "overlap":                    self.overlap,
            "gaussian":                   self.gaussian,
            "flip":                       self.flip,
            "verbose":                    self.verbose,
        }

        if export_path is not None:
            # if they've given a directory, dump as inference_config.json inside it
            if os.path.isdir(export_path):
                export_path = os.path.join(export_path, "inference_config.json")
            # if they gave a file name w/o .json, add it
            elif not export_path.lower().endswith(".json"):
                export_path = export_path + ".json"

            # ensure parent dir exists
            parent = os.path.dirname(export_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            with open(export_path, "w", encoding="utf-8") as _json_file:
                json.dump(_output_dict, _json_file, indent=4)
        else:
            print("export_path is None. Config not exported!")

    def import_from_json(self, import_path: str = None):
        assert import_path is not None and os.path.exists(import_path), (
            f"JSON file not found: {import_path}"
        )
        with open(import_path, "r", encoding="utf-8") as _file:
            _imported_data = json.load(_file)

        # assign back to every field
        self.model_path                = _imported_data.get("model_path")
        self.model_config              = _imported_data.get("model_config")
        self.output_dir                = _imported_data.get("output_dir")
        self.device                    = _imported_data.get("device")
        self.normalization_fingerprint = _imported_data.get("normalization_fingerprint")
        self.data_fingerprint          = _imported_data.get("data_fingerprint")
        self.patch_shape               = _imported_data.get("patch_shape")
        self.batch_size                = _imported_data.get("batch_size")
        self.overlap                   = _imported_data.get("overlap")
        self.gaussian                  = _imported_data.get("gaussian")
        self.flip                      = _imported_data.get("flip")
        self.verbose                   = _imported_data.get("verbose")
