import os
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 4 * 16000
feat_type = 'XXXXXX'
is_conti = False
conti_path = './XXXXXX'
conti_path = './XXXXXX'
is_pesq = True

# server parameter settings
json_dir = '/XXXXXX'
file_path = '/XXXXXX'
loss_dir = './XXXXXX'
batch_size = 2
epochs = 200
lr = 5e-4
model_best_path = './XXXXXX'
check_point_path = './XXXXXX'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)