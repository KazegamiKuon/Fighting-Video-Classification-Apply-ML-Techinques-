import torch as th
import math
import numpy as np
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
import torch
# Create class to extract video's features easily
# Class is based on the "extract content" of the "extract.ipynb" file.

#Tạo class để có 1 phương thức trích xuất đặc trưng một cách dễ dàng
# Class được viết dựa trên "extract content" của file "extract.ipynb"

class VideoExtracter(object):
    "Extract feature from video"
    def __init__(self,args):
        self.preprocess = Preprocessing(args.type)
        self.model = get_model(args)
        self.batch_size = args.batch_size
        self.l2_normalize = args.l2_normalize
        self.half_precision = args.half_precision

    def _pre_process_video(self,video):
        "video đầu vô là danh sách frame ở trạng thái bt"
        "video đầu ra định dạng là torch.Tensor với các frame đã được xử lý"
        temp = np.array(video)
        temp = torch.from_numpy(temp.astype('float32'))
        #chuyển vị các chiều
        temp = temp.permute(0, 3, 1, 2)
        return temp

    def predict(self,vid):
        with th.no_grad():
            data = self._pre_process_video(vid)
            if len(data.shape) > 3:
                video = data.squeeze()
                if len(video.shape) == 4:                    
                    video = self.preprocess(video)
                    n_chunk = len(video)
                    features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(self.batch_size)))
                    for i in range(n_iter):
                        min_ind = i * self.batch_size
                        max_ind = (i + 1) * self.batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        batch_features = self.model(video_batch)
                        if self.l2_normalize:
                            batch_features = F.normalize(batch_features, dim=1)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if self.half_precision:
                        features = features.astype('float16')
                    return features
            return None