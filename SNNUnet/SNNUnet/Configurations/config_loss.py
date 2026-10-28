""" 
Methods which implements losses. Losses also must be implemented in training.py
(Method __obtain_loss.py) --> more like a proper call of the loss function
"""
import json
import os

class LossConfigFromTorch(object):
    '''
    Class which implements simple loss which is avaliable in torch.
    Only name is requiered and that loss must be implemented in training.py
    '''
    '''
    loss_name: loss name must be a valid loss name. By the loss name the loss is instanced in training script.
    This is mandatory
    '''
    loss_name = "nll"

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += ("Loss\n")
        _retval += (f'loss_name: {LossConfigFromTorch.loss_name}\n')
        _retval += ("######################################\n")
        return _retval
    
    
    def export_to_json(self, export_path: str = None):
        """
        Export this config to JSON. If export_path is a directory
        (doesn't end in .json), writes to
        <export_path>/loss_from_torch_config.json.

        Args:
            * export_path, str, path to where the dict of the config elements will be stored. 

        """
        _data = {
            "loss_name": LossConfigFromTorch.loss_name
        }

        if export_path is None:
            print("export_path is None. Dict can not be exported!")
            return

        # If given a directory, append a filename
        if not export_path.endswith(".json"):
            assert os.path.exists(export_path), (
                f"Directory does not exist: {export_path}"
            )
            export_path = os.path.join(
                export_path, "loss_from_torch_config.json"
            )

        with open(export_path, "w") as _f:
            json.dump(_data, _f, indent=4)

    def import_from_json(self, import_path: str):
        """
        Load config from a JSON file previously written by export_to_json.
        Updates LossConfigFromTorch.loss_name.

        Args:
            * import_path, str, path to the "." json file which was exported
            by this class (method export_to_json)
        """
        assert import_path.endswith(".json") and os.path.exists(import_path), (
            f"JSON file not found: {import_path}"
        )

        with open(import_path, "r") as _f:
            _data = json.load(_f)

        LossConfigFromTorch.loss_name = _data["loss_name"]

class LossConfig_DCandCE(object):
    '''
    Class for specifying DCandCE loss
    '''
    '''
    loss_name: loss name must be a valid loss name. By the loss name the loss is instanced
     in training script. This is mandatory!
    '''
    loss_name = "DCandCE"

    # HERE GOES CUSTOM Flags or values required by the given loss. It can be literaly anything but
    # it must be specified.

    ##### This is for softDiceScore
    '''
    apply_nonlin: function (Callable) typicaly nnSoftmax(Dim=1). SoftDiceScore.py
    '''
    apply_nonlin = None

    '''
    batch_dice: boolean, see SoftDiceScore.py
    '''
    batch_dice = False

    '''
    do_bg: boolean, see SoftDiceScore.py
    '''
    do_bg = False

    '''
    smooth: float, see SoftDiceScore.py
    '''
    smooth = 1e-5

    '''
    clip_tp: float,  see SoftDiceScore.py
    '''
    clip_tp = None

    ##### Main config of DC and CE
    '''
    weight_dice, float, weight put on the dice part of the loss. 
    Remeber: Loss = weight_dice*DC + weight_ce*CE
    '''
    weight_dice = 1.0

    '''
    weight_ce, float, weight put on the dice part of the loss. 
    Remeber: Loss = weight_dice*DC + weight_ce*CE
    '''
    weight_ce = 1.0

    '''
    ignore_label: labels to be ignored when calculating the loss function. It is set
    to default value of torch.nn.CrossEntropyLoss
    '''
    ignore_label = -100

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += ("Loss\n")
        _retval += (f'loss_name: {LossConfig_DCandCE.loss_name}\n' +
                    f'apply_nonlin: {LossConfig_DCandCE.apply_nonlin.__class__.__name__}\n' +
                    f'batch_dice: {LossConfig_DCandCE.batch_dice}\n' +
                    f'do_bg: {LossConfig_DCandCE.do_bg}\n' + 
                    f'smooth: {LossConfig_DCandCE.smooth}\n' +
                    f'clip_tp: {LossConfig_DCandCE.clip_tp}\n' +
                    f'weight_dice: {LossConfig_DCandCE.weight_dice}\n' + 
                    f'ignore_label: {LossConfig_DCandCE.ignore_label}\n' +
                    f'weight_ce: {LossConfig_DCandCE.weight_ce}\n')
        _retval += ("######################################\n")
        return _retval 

    def export_to_json(self, 
                       export_path: str = None):
        """
        Export this loss config to JSON. If "export_path" is a directory,
        writes to <export_path>/loss_dcandce_config.json.

        Args:
            * export_path, str, path to where the dict of the config elements will be stored. 

        """
        _data = {
            "loss_name": LossConfig_DCandCE.loss_name,
            "apply_nonlin": (
                LossConfig_DCandCE.apply_nonlin.__class__.__name__
                if LossConfig_DCandCE.apply_nonlin is not None else None
            ),
            "batch_dice": LossConfig_DCandCE.batch_dice,
            "do_bg": LossConfig_DCandCE.do_bg,
            "smooth": LossConfig_DCandCE.smooth,
            "clip_tp": LossConfig_DCandCE.clip_tp,
            "weight_dice": LossConfig_DCandCE.weight_dice,
            "weight_ce": LossConfig_DCandCE.weight_ce,
            "ignore_label": LossConfig_DCandCE.ignore_label
        }

        if export_path is None:
            print("export_path is None. Dict can not be exported!")
            return

        # If given a directory, append a filename
        if not export_path.endswith(".json"):
            assert os.path.exists(export_path), (
                f"Directory does not exist: {export_path}"
            )
            export_path = os.path.join(export_path, "loss_dcandce_config.json")

        with open(export_path, "w") as _f:
            json.dump(_data, _f, indent=4)

    def import_from_json(self, import_path: str):
        """
        Load config from a JSON file previously written by export_to_json.
        Note: apply_nonlin will be set to the *name* of the function, not
        the callable itself.

        Args:
            * import_path, str, path to the ".json" file which was exported
            by this class (method export_to_json)
        """
        assert import_path.endswith(".json") and os.path.exists(import_path), (
            f"JSON file not found: {import_path}"
        )

        with open(import_path, "r") as _f:
            _data = json.load(_f)

        LossConfig_DCandCE.loss_name     = _data["loss_name"]
        LossConfig_DCandCE.apply_nonlin  = _data["apply_nonlin"]
        LossConfig_DCandCE.batch_dice    = _data["batch_dice"]
        LossConfig_DCandCE.do_bg         = _data["do_bg"]
        LossConfig_DCandCE.smooth        = _data["smooth"]
        LossConfig_DCandCE.clip_tp       = _data["clip_tp"]
        LossConfig_DCandCE.weight_dice   = _data["weight_dice"]
        LossConfig_DCandCE.weight_ce     = _data["weight_ce"]
        LossConfig_DCandCE.ignore_label  = _data["ignore_label"]

class LossConfigDeepSupervision(object):
    '''
    Class for specifying DS loss
    '''
    '''
    name: First, as always Name
    '''
    loss_name = "dsLoss"

    '''
    loss_function: config of the loss which is going to be applied, for instance
    it can be _config = LossConfig_DCandCE() which will then build the loss
    from the givne config
    '''
    loss_function: None

    '''
    cascade_number: number of downsamplings output shapes (upsamplings). The real number
    of applied losses will always be this number -1
    '''
    cascade_number = 5

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += ("DS Loss\n")
        _retval += (f'loss_name: {LossConfigDeepSupervision.loss_name}\n' +
                    f'cascade_number: {LossConfigDeepSupervision.cascade_number}\n')
        if LossConfigDeepSupervision.loss_function is not None:
            _retval += f"loss_function: {LossConfigDeepSupervision.loss_function.__class__.__name__}\n"
        else:
            f'loss_function: {LossConfigDeepSupervision.loss_function}\n'
        _retval += ("######################################\n")
        return _retval
    
    def export_to_json(self, export_path: str = None):
        """
        Export this config to a JSON file. If export_path is a directory
        (doesn't end in .json), writes to <export_path>/loss_ds_config.json.

        Args:
            * export_path, str, path to where the dict of the config elements will be stored. 
        """
        _data = {
            "loss_name": LossConfigDeepSupervision.loss_name,
            "cascade_number": LossConfigDeepSupervision.cascade_number,
            "loss_function": (
                LossConfigDeepSupervision.loss_function.__class__.__name__
                if LossConfigDeepSupervision.loss_function is not None else None
            )
        }

        # Fix naming
        if export_path is None:
            print("export_path is None. Dict can not be exported!")
            return

        # if it's a dir, append a filename
        if not export_path.endswith(".json"):
            assert os.path.exists(export_path), (
                f"Directory for export does not exist: {export_path}"
            )
            export_path = os.path.join(export_path, "loss_ds_config.json")

        with open(export_path, "w") as _f:
            json.dump(_data, _f, indent=4)

    def import_from_json(self, 
                         import_path: str):
        """
        Load config from a JSON file previously written by export_to_json.
        Note: this will set loss_function to the *name* of the class,
        not the actual callable.

        Args:
            * import_path, str, path to the ".json" file which was exported
            by this class (method export_to_json)
        """
        assert import_path.endswith(".json") and os.path.exists(import_path), (
            f"JSON file not found: {import_path}"
        )

        with open(import_path, "r") as _f:
            _data = json.load(_f)

        LossConfigDeepSupervision.loss_name = _data["loss_name"]
        LossConfigDeepSupervision.cascade_number = _data["cascade_number"]
        LossConfigDeepSupervision.loss_function = _data["loss_function"]