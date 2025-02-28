import os
from mri_localisation import *

def save_array(output_dir, array):
    for i in range(array.shape[0]):
        img = array[i,:,:]
        img = (img / np.max(img)) * 255
        img = img.astype(np.uint8)
        cv2.imwrite(output_dir + output_dir.split("/")[-2] + "_slice_" + str(i) + ".png", img)


AD_folder = os.walk(os.getcwd() + "/ADNI/AD")
MCI_folder = os.walk(os.getcwd() + "/ADNI/MCI")
CN_folder = os.walk(os.getcwd() + "/ADNI/CN")

# Add all files to list
endings = ["AD", "MCI", "CN"]
for ending in endings:
    output_path = "localisation_outputs/" + ending +"/"
    folder = os.walk(os.getcwd() + "/ADNI/" + ending)
    MRI_files = []
    for root, dirs, files in folder:
        for file in files:
            if file.endswith(".nii"):
                MRI_files.append(os.path.join(root, file))
    bad_localisations = []
    bad_reads = []
    for file in MRI_files:
        print("localising file " + file)
        try:
            localiser = MRILocaliser(file)
            try:
                localiser.localise()
                if localiser.data_array.shape[0] != 20:
                    print("Something weird with file " + file)
                    bad_localisations.append(file)
            except:
                print("Error localising file " + file)
                bad_localisations.append(file)
        except:
            print("Error reading file " + file)
            bad_reads.append(file)
        print("dims = " + str(localiser.data_array.shape))
        # scan_name = os.path.basename(file).split(".")[0]
        # output_dir = output_path + scan_name + "/"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # save_array(output_dir, localiser.data_array)

    with open(output_path + "bad_localisations.txt", "w") as f:
        for file in bad_localisations:
            f.write(file + "\n")
    with open(output_path + "bad_reads.txt", "w") as f:
        for file in bad_reads:
            f.write(file + "\n")