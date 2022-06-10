import argparse
import json
from keras.models import model_from_json
import numpy as np

from keras.engine.functional import Functional
from keras.engine.input_layer import InputLayer

def get_arch_layers(arch_layers):
    """Convert all layers in a single list, including Functional layers.

    Args:
        arch_layers (list): architecture layers.
    """
    layers = []

    while arch_layers != []:
        if (arch_layers[0]['class_name'] == 'Functional'):
            sub_layers = arch_layers.pop(0)['config']['layers']

            if (sub_layers[0]['class_name'] == 'InputLayer'):
                sub_layers.pop(0)

            arch_layers = sub_layers + arch_layers
        else:
            layers.append(arch_layers.pop(0))

    return layers

def get_model_layers(model_layers):
    """Convert all layers in a single list, including Functional layers.

    Args:
        model_layers (list): model layers.
    """
    layers = []

    while model_layers != []:
        if (isinstance(model_layers[0], Functional)):
            sub_layers = model_layers.pop(0).layers

            if (isinstance(sub_layers[0], InputLayer)):
                sub_layers.pop(0)

            model_layers = sub_layers + model_layers
        else:
            layers.append(model_layers.pop(0))

    return layers

def convert_model_top(args):
    """Convert input JSON file to a keras model.

    Args:
        args (Sequence[str]): parsed arguments from console.
    """
    # Read the NN architecture
    arch = open(args.architecture).read()
    # Parsing the model from JSON format
    model = model_from_json(arch)
    # Load weights in H5 format.
    model.load_weights(args.weights)
    # Compile the Keras model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    # Get JSON file as a Python dictionary
    arch = json.loads(arch)

    # Empty the content of the model into an output file used by keras2cpp
    with open(args.output, 'w') as fout:
        input_layer = 0
        arch_layers = get_arch_layers(arch['config']['layers'])
        model_layers = get_model_layers(model.layers)
        activations = 0

        print(f'layers {len(arch_layers)}')
        fout.write(f'layers {len(arch_layers)}\n')

        for ind, l in enumerate(arch_layers):
            if args.verbose:
                print(ind + activations, l)

            fout.write(f'layer {ind + activations} {l["class_name"]}\n')

            print(f'{ind + activations} {l["class_name"]}')

            if l['class_name'] == 'InputLayer':
                input_layer += 1
            elif l['class_name'] == 'Conv2D':
                W = model_layers[ind - input_layer].get_weights()[0]
                w_rows, w_cols, w_depth, w_batch = W.shape
                W = W.reshape((w_batch, w_depth, w_rows, w_cols))

                if args.verbose:
                    print(W.shape)

                fout.write(f'{W.shape[0]} {W.shape[1]} {W.shape[2]} {W.shape[3]} {l["config"]["padding"]}\n')
                print(f'{W.shape[0]} {W.shape[1]} {W.shape[2]} {W.shape[3]} {l["config"]["padding"]}')

                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i, j, k]) + '\n')

                fout.write(f'{model_layers[ind - input_layer].get_weights()[1]}\n')
            elif l['class_name'] == 'MaxPooling2D':
                fout.write(f'{l["config"]["pool_size"][0]} {l["config"]["pool_size"][1]}\n')
                print(f'{l["config"]["pool_size"][0]} {l["config"]["pool_size"][1]}')
            elif l['class_name'] == 'Flatten':
                print(l['config']['name'])
            elif l['class_name'] == 'Dense':
                W = model_layers[ind - input_layer].get_weights()[0]

                if args.verbose:
                    print(W.shape)

                fout.write(f'{W.shape[0]} {W.shape[1]}\n')
                print(f'{W.shape[0]} {W.shape[1]}')

                for w in W:
                    fout.write(f'{w}\n')

                fout.write(f'{model_layers[ind - input_layer].get_weights()[1]}\n')

            if 'activation' in l['config']:
                activations += 1
                print(f'{ind + activations} Activation')
                print(f'{l["config"]["activation"]}')
                fout.write(f'layer {ind + activations} Activation\n')
                fout.write(f'{l["config"]["activation"]}\n')

                if args.verbose:
                    print(l["config"]["activation"])

        fout.write(f'layer {ind + activations} End\n')
        print(f'{ind + activations} End')

if __name__ == '__main__':
    np.random.seed(1337)
    np.set_printoptions(threshold=np.inf)

    # Parsing console arguments
    parser = argparse.ArgumentParser(
        description='This is a simple script to dump Keras model into simple format suitable for porting into pure C++ model')

    parser.add_argument('-a', '--architecture',
                        help="JSON with model architecture", required=True)
    parser.add_argument('-w', '--weights',
                        help="Model weights in HDF5 format", required=True)
    parser.add_argument(
        '-o', '--output', help="Ouput file name", required=True)
    parser.add_argument('-v', '--verbose', help="Verbose", required=False)
    args = parser.parse_args()
    args.verbose = args.verbose == '1'

    print(f'Read architecture from {args.architecture}')
    print(f'Read weights from {args.weights}')
    print(f'Writing to {args.output}')

    convert_model_top(args)
