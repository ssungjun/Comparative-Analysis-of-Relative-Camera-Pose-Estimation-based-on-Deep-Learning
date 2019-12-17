from util import *
import numpy as np
from argparser import give_parser
import cv2
#import Image

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return buf#cv2.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def visualize(args, best_type='trans'):#['rot', 'ang', 'trans', 'total']
    folder_name = args.net + '_' + args.dataset + '_' + args.seven_opt + '_' + best_type + '_eval'
    eval_folder = os.path.join('output', folder_name)
    graph_folder = '_'.join(eval_folder.split('_')[:-1])+'_graph'

    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)

    dataset_name = eval_folder.split('_')[1]
    seven_opt = eval_folder.split('_')[2]

    if dataset_name[0] == 'k':
        mass = [0, 4070, 5660, 6860]#test
    elif dataset_name[0] == '3':
        mass = [0, 897, 2299, 4476, 5676, 6695, 7282, 8138, 9586, 10569, 11299, 12081, 13417, 14656, 15500, 16871, 17258, 18644, 19232, 20424, 22371, 23118, 24396, 25275, 26216]
    else:
        if seven_opt[0] == 't':
            mass = [0, 999, 1998, 2997, 3996, 4995, 5994, 6993, 7992, 8991, 9990, 10989, 11988, 12987, 13986, 14985, 15984, 16483, 16982]

    for seq in range(len(mass) - 1):
        start = mass[seq]
        end = mass[seq + 1]
        gt_transform = np.eye(4)
        pred_transform = np.eye(4)
        gt_cartesian = [[0, 0, 0]]
        pred_cartesian = [[0, 0, 0]]
        for trans in range(start,end):
            with open(os.path.join(eval_folder, '%06d_eval.txt'%(trans)), 'rb') as f:
                data = f.readlines()
                for i in range(4):
                    temp_data = data[i].decode().split(':')[1].split(',')
                    if i == 0:
                        gt_quat = [float(temp_data[0]), float(temp_data[1]), float(temp_data[2]), float(temp_data[3])]
                    elif i == 1:
                        pred_quat = [float(temp_data[0]), float(temp_data[1]), float(temp_data[2]), float(temp_data[3])]
                    elif i == 2:
                        gt_trans = [float(temp_data[0]), float(temp_data[1]), float(temp_data[2])]
                    else:
                        pred_trans = [float(temp_data[0]), float(temp_data[1]), float(temp_data[2])]
            relative_gt_transform = quaternion_translation_to_transform(gt_quat, gt_trans)
            relative_pred_transform = quaternion_translation_to_transform(pred_quat, pred_trans)
            gt_transform = np.matmul(gt_transform,relative_gt_transform)
            pred_transform = np.matmul(pred_transform, relative_pred_transform)
            gt_cartesian.append([gt_transform[0,-1],gt_transform[1,-1],gt_transform[2,-1]])
            pred_cartesian.append([pred_transform[0, -1], pred_transform[1, -1], pred_transform[2, -1]])
        gt_array = np.array(gt_cartesian)
        pred_array = np.array(pred_cartesian)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')  # Axe3D object
        ax.plot(gt_array[:, 0], gt_array[:, 1], gt_array[:, 2], alpha=0.1, marker='.')
        ax.plot(pred_array[:, 0], pred_array[:, 1], pred_array[:, 2], alpha=0.1, marker='.')
        plt.title("3D Trajectory")
        plt.legend(('GT', 'Predicted'), loc='upper right')
        plt.savefig(os.path.join(graph_folder, '%d_3Dgraph.png'%(seq)))

        fig1 = plt.figure(figsize=(10, 5))
        ax1 = fig1.add_subplot(111)  # Axe3D object
        ax1.plot(gt_array[:, 0], gt_array[:, 1], alpha=0.5, marker='.')
        ax1.plot(pred_array[:, 0], pred_array[:, 1], alpha=0.5, marker='.')
        plt.title("2D Trajectory")
        plt.legend(('GT', 'Predicted'), loc='upper right')
        plt.savefig(os.path.join(graph_folder, '%d_2Dgraph.png' % (seq)))

if __name__ == '__main__':
    visualize(give_parser())