""" Config of the VNet"""
import json
import os

class ModelConfig:
    '''
    Class for specifying augmentation on the training set data
    '''
    '''
    model_name: name of the model, string, based on this model name the model is
    chosen for the training
    '''
    model_name = None

    '''
    output_classes, integer, number of output classes of the model
    '''
    number_of_output_classes = 7

    '''
    elu: boolean, it flags should elu be used in some models
    '''
    elu = True

    '''
    nll: boolean, it flags type of softmax at the end in some models. See model implementation
    for exact meaning
    '''
    nll = True

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += ("Model\n")
        _retval += (f'model_name: {ModelConfig.model_name}\n' +
                    f'output_classes: {ModelConfig.number_of_output_classes}\n' +
                    f'elu: {ModelConfig.elu}\n' +
                    f'nll: {ModelConfig.nll}\n')
        _retval += ("######################################\n")
        return _retval

    def export_to_json(
        self,
        export_path: str = None):
        """
        Method which exports the dict to the json. The default "VNet_fingerprint.json" 
        as the name is added only if the path already does not end with ".json". 
        In that case I assume that the exact file is provided.

        Args:
            * export_path, str, path to where the dict of the config elements will be stored. 
        """
        # Build dict:
        _output_dict = {"model_name": ModelConfig.model_name, 
                        "output_classes": ModelConfig.output_classes, 
                        "elu": ModelConfig.elu,
                        "nll": ModelConfig.nll}

         # In case that we want to export it
        if export_path is not None:
            # Modify name if dir is provided
            if not export_path.endswith(".json"):
                assert os.path.exists(export_path), (f"Path for the export does not exists. "
                                                     f"Please check path: {export_path}!")
                export_path = os.path.join(export_path, "VNet_fingerprint.json")

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
        ModelConfig.output_classes = _imported_data["output_classes"]
        ModelConfig.elu = _imported_data["elu"]
        ModelConfig.nll = _imported_data["nll"]
