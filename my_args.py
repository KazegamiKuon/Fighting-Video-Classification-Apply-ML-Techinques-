#create args class
class MyArgs(object):
    def __init__(
            self,
            csv = 'test_input.csv',
            model_type = '2d',
            batch_size = 64,
            num_decoding_thread = 4,
            resolution = 112,
            framerate = 24,
            half_precision = None,
            l2_normalize = None,
            scale_time = None,
            resnext101_model_path='./model_feature_extraction/resnext101.pth',
        ):
        self.csv = csv
        self.type = model_type
        self.batch_size = batch_size
        self.num_decoding_thread = num_decoding_thread
        self.resolution = resolution
        self.framerate = framerate
        self.half_precision = half_precision
        self.l2_normalize = l2_normalize
        self.scale_time = scale_time
        self.resnext101_model_path= resnext101_model_path