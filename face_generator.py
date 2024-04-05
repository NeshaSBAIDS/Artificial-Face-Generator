# Import necessary libraries
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization."""
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim * 8 * 2 * 2)
        self.t_conv1 = deconv(conv_dim * 8, conv_dim * 4)
        self.t_conv2 = deconv(conv_dim * 4, conv_dim * 2)
        self.t_conv3 = deconv(conv_dim * 2, conv_dim)
        self.t_conv4 = deconv(conv_dim, 3, batch_norm=False)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, self.conv_dim * 8, 2, 2)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = self.t_conv4(out)
        out = F.tanh(out)
        return out

# Instantiate the Generator and load the pre-trained weights
g_conv_dim = 32
z_size = 100
G = Generator(z_size=z_size, conv_dim=g_conv_dim)
G_state_dict = torch.load(r'/Users/nesha/Downloads/Winter-intern-project/G_checkpoint.pth', map_location=torch.device('cpu'))
G.load_state_dict(G_state_dict)
G.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])  # Specify POST as the allowed method
def generate():
    # Generate image using GAN
    z = np.random.uniform(-1, 1, size=(1, z_size))
    z = torch.from_numpy(z).float()
    fake_image = G(z)

    # Post-process the generated image (you may need to adjust this based on your GAN output)
    fake_image = fake_image.detach().cpu().numpy().squeeze()
    fake_image = np.transpose(fake_image, (1, 2, 0))
    fake_image = ((fake_image + 1) * 255 / 2).astype(np.uint8)
    
    # Save the generated image
    pil_image = Image.fromarray(fake_image)
    image_path = r'/Users/nesha/Downloads/Winter-intern-project/templates/generated_image1.png'
    pil_image.save(image_path)

    return render_template('result.html', image_path=image_path)

# Add a route to serve static files (images)
@app.route('/templates/<filename>')
def send_image(filename):
    return send_from_directory("templates", filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
