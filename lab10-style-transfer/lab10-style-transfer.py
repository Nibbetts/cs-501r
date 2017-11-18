
import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize

from skimage.io import imsave
import matplotlib.pyplot as plt

sess = tf.Session()

opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'style.png', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'content.png', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]
# ops[8] = content, ops[0,2,4,7,10] are style layers.

content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

#
# --- construct your cost function here
def gram_array(array): # <--- I had to get help on this from code examples...
    matrix = tf.reshape(array, shape=[-1,array.shape[-1]])
    return tf.matmul(tf.transpose(matrix), matrix)

def gram_tensor(tensor): # <--- I had to get help on this from code examples...
    matrix = tf.reshape(tensor, shape=[-1,tensor.get_shape().as_list()[-1]])
    return tf.matmul(tf.transpose(matrix), matrix)

alpha = 1
beta = 5e3
style_ind = [0,2,4,7,10]

content_acts = content_acts[8]
style_acts = [style_acts[i] for i in style_ind]
content_layer = ops[8]
style_layers = [ops[i] for i in style_ind]

l_content = 0.5 * tf.reduce_sum(tf.square(content_layer - content_acts), name="l_content")

N_l = [l.shape[-1] for l in style_acts]
M_l = [l.shape[1] * l.shape[2] for l in style_acts]

gram_original = [gram_array(l) for l in style_acts]
gram_gen = [gram_tensor(l) for l in style_layers] # NOTE: from here on out I don't know how to vectorize it!
E_l = np.array([1.0/(4 * N_l[l]**2 * M_l[l]**2) * tf.reduce_sum(tf.square(gram_original[l] - gram_gen[l]))
       for l in range(len(style_acts))])
w_l = np.array([1.0/len(style_layers)] * len(style_layers))
l_style = tf.reduce_sum([E_l[l] * w_l[l] for l in range(len(style_layers))])
l_total = alpha * l_content + beta * l_style

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)
train_step = (tf.train.AdamOptimizer(learning_rate=0.1,name='adam_optimizer')
              .minimize(l_total,var_list=[opt_img]))

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
sess.run( opt_img.assign( content_img ))

# --- place your optimization loop here
print "STEP\tLOSS\t\tSTYLE LOSS\tCONTENT LOSS"
for i in xrange(6000):
    _, t, s, c = sess.run([train_step, l_total, l_style, l_content])
    if i%100 == 0:
        print i, '\t', t, '\t', s, '\t', c
        img = sess.run(opt_img)
        img = tf.maximum(0.0,img)
        img = tf.minimum(255.0,img)
        sess.run(opt_img.assign(img))

stylized = sess.run(opt_img)[0]
stylized = np.maximum(0.0,stylized)
stylized = np.minimum(255.0,stylized)

imsave("stylized.png",stylized*2/255.0-1)
#plt.imshow(stylized)


# None of the generated images seem to be very pretty :(
# ... but at least it works.....
# NOTE that I think the weirdness of my results has something to do with the way I'm clipping, though I'm not sure how else to do it.
