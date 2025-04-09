import numpy as np

# Network dimensions
INPUT_DIM = 5
HIDDEN_DIM = 32
OUTPUT_DIM = 3

# Generate random weights with small values
def generate_weights(rows, cols):
    return np.random.uniform(0, 0.1, (rows, cols))

# Generate biases with small values
def generate_biases(size):
    return np.random.uniform(0,0.1, size)

# Generate the weights and biases
weights1 = generate_weights(INPUT_DIM, HIDDEN_DIM)
bias1 = generate_biases(HIDDEN_DIM)
weights2 = generate_weights(HIDDEN_DIM, HIDDEN_DIM)
bias2 = generate_biases(HIDDEN_DIM)
weights3 = generate_weights(HIDDEN_DIM, OUTPUT_DIM)
bias3 = generate_biases(OUTPUT_DIM)

# Write to header file
with open('trained_weights_dummy.h', 'w') as f:
    f.write('#ifndef TRAINED_WEIGHTS_DUMMY_H\n')
    f.write('#define TRAINED_WEIGHTS_DUMMY_H\n\n')
    
    f.write('#define INPUT_DIM 5\n')
    f.write('#define HIDDEN_DIM 32\n')
    f.write('#define OUTPUT_DIM 3\n\n')
    
    # Write weights1
    f.write('const float weights1[INPUT_DIM][HIDDEN_DIM] = {\n')
    for i in range(INPUT_DIM):
        f.write('    {')
        f.write(', '.join([f'{x:.6f}f' for x in weights1[i]]))
        f.write('},\n')
    f.write('};\n\n')
    
    # Write bias1
    f.write('const float bias1[HIDDEN_DIM] = {\n    ')
    f.write(', '.join([f'{x:.6f}f' for x in bias1]))
    f.write('\n};\n\n')
    
    # Write weights2
    f.write('const float weights2[HIDDEN_DIM][HIDDEN_DIM] = {\n')
    for i in range(HIDDEN_DIM):
        f.write('    {')
        f.write(', '.join([f'{x:.6f}f' for x in weights2[i]]))
        f.write('},\n')
    f.write('};\n\n')
    
    # Write bias2
    f.write('const float bias2[HIDDEN_DIM] = {\n    ')
    f.write(', '.join([f'{x:.6f}f' for x in bias2]))
    f.write('\n};\n\n')
    
    # Write weights3
    f.write('const float weights3[HIDDEN_DIM][OUTPUT_DIM] = {\n')
    for i in range(HIDDEN_DIM):
        f.write('    {')
        f.write(', '.join([f'{x:.6f}f' for x in weights3[i]]))
        f.write('},\n')
    f.write('};\n\n')
    
    # Write bias3
    f.write('const float bias3[OUTPUT_DIM] = {\n    ')
    f.write(', '.join([f'{x:.6f}f' for x in bias3]))
    f.write('\n};\n\n')
    
    f.write('#endif // TRAINED_WEIGHTS_DUMMY_H\n')

print("Generated trained_weights_dummy.h successfully!") 