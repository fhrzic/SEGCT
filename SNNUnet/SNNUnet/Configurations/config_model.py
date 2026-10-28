"""Config class for SNNUnet"""
import json
import os
import sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from Models.SNNUnet.generate_SNNUnet_fingerprint import GenerateSNNUnetFingerprint

class ModelConfig:
    """
    Class for specfying all necessary propertis for building SNNUnet
    """
    '''
    model_name: str, named of the model based on this model name the model is
    chosen for the training.
    '''
    model_name = "SNNUnet"

    '''
    number_of_channels: int, number of input channels to the SNNUnet.
    '''
    number_of_input_channels = 1

    '''
    number_of_output_classes, int, number of output classes of the model.
    '''
    number_of_output_classes = 16

    '''
    input_shape: tuple, this is the input shape of the SNNUnet. It can be calucalted if the 
    fingerprint is provided. Defualt is 128, 128, 128
    '''
    input_shape = (128, 128, 128)

    '''
    input_spacing: tuple, input spacing of the input volumes. Default is 1,1,1
    '''
    input_spacing = (1,1,1)

    '''
    z_last_channel:bool, if the z is last channel (True), or first channel (False). 
    Z is offplane plane.
    '''
    z_last_channel = True

    '''
    max_batch_size:int, what is the biggest batch that can be processed on gpu
    '''
    max_batch_size = 2

    '''
    output_shapes, list, list representing output shapes of the model 
    (needed for deep supervision loss)
    '''
    output_shapes = None

    '''
    output_spacings, list, list of the output spacings 
    (decorative I think but might be usefull in some conetext)
    '''
    output_spacings = None

    '''
    output_auxiliary_predictions, bool, if set to true the model will output auxiliary 
    predictions (predictions after every up block)
    '''
    output_auxiliary_predictions = True

    '''
    device_limit:int, in case that we dont want to use the whole device, we can provide
    number of megabytes that we want to alocate for batch size 2 forward barckward pass
    on the GPU. One of the ways how we can increase the batch size. None means we use the 
    whole device.
    '''
    device_limit = None

    '''
    mixed_precision, bool, if set to true, the whole model will be done in mixed precision
    '''
    mixed_precision = False

    def __init__(
        self,
        path_to_data_fingerprint: str = None,
        device: str = None,
        device_limit: int = None,
        mixed_precision: bool = False):
        """
        Init method to the fingerprint file. 

        If the both path_to_data_fingerprint and device are provided, all the 
        parameters will be computated on its own.

        Args:
            * path_to_data_fingerprint, str, path to the data fingerprint 
            (generated in DatasetFingerprint).
            * device, str, name of the device which is going to be used.
            * device_limit, int, set this numver if only part of the GPU should
            be used. It is defined in MB.
            * mixed_precision, bool, if set to True, the model will be used in mixed
            precision enabeling bigger patch sizes. CAUTION: SAME MUST BE APPLIED
            DURING TRAINING (otherwise we risk that model will not fit on the GPU
            in full precision).

        """
        # Auto computation of all parameters
        if path_to_data_fingerprint is not None and device is not None:
            # Generate fingerprint
            _generate_fingerprint = GenerateSNNUnetFingerprint(
                dataset_fingerprint_path = path_to_data_fingerprint,
                device = device,
                device_limit = device_limit,
                mixed_precision = mixed_precision)

            # Obtain data
            _fingerprint_dict = _generate_fingerprint.obtain_fingerprint(export_path = None)
            # Transfer data to class
            ModelConfig.number_of_input_channels = _fingerprint_dict["number_of_input_channels"]
            ModelConfig.max_batch_size = _fingerprint_dict["max_batch_size"]
            ModelConfig.number_of_output_classes = _fingerprint_dict["number_of_output_classes"]
            ModelConfig.input_shape = _fingerprint_dict["input_patch_shape"]
            ModelConfig.input_spacing = _fingerprint_dict["input_spacing"]
            ModelConfig.z_last_channel = _fingerprint_dict["z_last_channel"]
            ModelConfig.output_shapes = _fingerprint_dict["output_shapes"]
            ModelConfig.output_spacings = _fingerprint_dict["output_spacings"]
            ModelConfig.device_limit = device_limit
            ModelConfig.mixed_precision = mixed_precision

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += ("Model SNNUnet\n")
        _retval += (f'model_name: {ModelConfig.model_name}\n' +
                   f'number_of_channels : {ModelConfig.number_of_input_channels}\n' +
                   f'number_of_output_classes : {ModelConfig.number_of_output_classes}\n' +
                   f'input_shape : {ModelConfig.input_shape}\n' +
                   f'input_spacing : {ModelConfig.input_spacing}\n' +
                   f'z_last_channel : {ModelConfig.z_last_channel}\n' +
                   f'max_batch_size : {ModelConfig.max_batch_size}\n' +
                   f'output_shapes : {ModelConfig.output_shapes}\n' +
                   f'output_spacings : {ModelConfig.output_spacings}\n' +
                   f'output_auxiliary_predictions: {ModelConfig.output_auxiliary_predictions}\n' +
                   f'device_limit : {ModelConfig.device_limit}\n' + 
                   f'mixed_precision : {ModelConfig.mixed_precision}\n')
        _retval += ("######################################\n")
        # Return
        return _retval

    def __iter__(
        self):
        '''
        Returns only parameters required by the model if the config is called as dict(model_config)
        '''
        yield "input_shape", self.input_shape
        yield "input_spacing", self.input_spacing
        yield "number_of_channels", self.number_of_input_channels
        yield "z_last_channel", self.z_last_channel
        yield "number_of_output_classes", self.number_of_output_classes
        yield "output_auxiliary_predictions", self.output_auxiliary_predictions

    def export_to_json(
        self,
        export_path: str = None):
        """
        Method which exports the dict to the json. The default "SNNUnet_fingerprint.json" 
        as the name is added only if the path already does not end with ".json". 
        In that case I assume that the exact file is provided.

        Args:
            * export_path, str, path to where the dict of the config elements will be stored. 
        """
        # Build dict:
        _output_dict = {"model_name": ModelConfig.model_name, 
                        "number_of_channels": ModelConfig.number_of_input_channels, 
                        "number_of_output_classes": ModelConfig.number_of_output_classes,
                        "input_shape": ModelConfig.input_shape,
                        "input_spacing": ModelConfig.input_spacing,
                        "z_last_channel": ModelConfig.z_last_channel,
                        "max_batch_size": ModelConfig.max_batch_size,
                        "output_shapes": ModelConfig.output_shapes,
                        "output_spacings": ModelConfig.output_spacings, 
                        "output_auxiliary_predictions": ModelConfig.output_auxiliary_predictions,
                        "device_limit": ModelConfig.device_limit,
                        "mixed_precision": ModelConfig.mixed_precision}

         # In case that we want to export it
        if export_path is not None:
            # Modify name if dir is provided
            if not export_path.endswith(".json"):
                assert os.path.exists(export_path), (f"Path for the export does not exists. "
                                                     f"Please check path: {export_path}!")
                export_path = os.path.join(export_path, "SNNUnet_fingerprint.json")

            # Dump it
            with open(export_path, "w") as _json_file:
                json.dump(_output_dict, _json_file, indent=4)
        else:
            print("export_path is None. Dict can not be exported!")

    def import_from_json(
        self,
        import_path: str = None):
        """
        Method which imports data from the provided import path and preloads whole
        class with imported data. 

        Args:
            * import_path, str, path to the "." json file which was exported
            by this class (method export_to_json)
        """
        # Check if file exists
        assert os.path.exists(import_path), (f"Path for the export does not exists. "
                                        f"Please check path: {import_path}!")

        # Read json to a dict
        with open(import_path, 'r') as _file:
            _imported_data = json.load(_file)
        # Transfer to variables
        ModelConfig.model_name = _imported_data["model_name"]
        ModelConfig.number_of_channels = _imported_data["number_of_channels"]
        ModelConfig.number_of_output_classes = _imported_data["number_of_output_classes"]
        ModelConfig.input_shape = _imported_data["input_shape"]
        ModelConfig.input_spacing = _imported_data["input_spacing"]
        ModelConfig.z_last_channel = _imported_data["z_last_channel"]
        ModelConfig.max_batch_size = _imported_data["max_batch_size"]
        ModelConfig.output_shapes = _imported_data["output_shapes"]
        ModelConfig.output_spacings = _imported_data["output_spacings"]
        ModelConfig.output_auxiliary_predictions = _imported_data["output_auxiliary_predictions"]
        ModelConfig.device_limit = _imported_data["device_limit"]
        ModelConfig.mixed_precision = _imported_data["mixed_precision"]
        