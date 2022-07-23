import os

folder= '/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1'
blurry_folder=os.path.join(folder,'blurry')
blurry_files = os.listdir(blurry_folder)[:3]
positions_folder=os.path.join(folder,'positions')
positions_files = os.listdir(positions_folder)[:3]

for blurry, position in zip(blurry_files, positions_files):
    blurry_filename = os.path.join(blurry_folder, blurry)
    positions_filename = os.path.join(positions_folder, position)
    print(blurry_filename, positions_filename)
    os.system('python test_restoration_from_positions.py -b ' + blurry_filename +  ' -p ' + positions_filename + ' -rf 0.6')