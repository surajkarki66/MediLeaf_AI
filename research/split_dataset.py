import splitfolders

input_folder = "./dataset_v2"
output_folder = "./dataset_v2_4_2_4"
ratio = (.4, .2, .4) #(train, val, test)


splitfolders.ratio(input_folder, output_folder,
					seed=42, ratio=ratio,
					group_prefix=None)
					
