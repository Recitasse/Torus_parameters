
# Torus_parameters
Determine torus parameters from dataset by AI

## Create your own datasets
In the file *model_train.py* you can change the datasets parametrisation.
You can then, create your own datasets to train your models with your own configurations :

**The offset parameters [x0low, x0high]**
x0low = -3
x0high = 3
y0low = -3
y0high = 3
z0low = -3
z0high = 3

Offsets are generated with those delimitations, the random distribution is the uniform distribution, to get an even distribution (the same *amounts* of event to not bias the model).

**Radius of the tori**
r1low = 1.5 
r1high = 3
r2low = 0.01
r2high = 1.5

Same principle, excpet that the radius cannot be negativs and the maximum "r2high" lower or equal to "r1low" ! 

**Orientation euler's angles : Euler Z-X-Z convention (al, be, ga)**
allow = -2*np.pi
alhigh = 2*np.pi
below = -2*np.pi
behigh = 2*np.pi
galow = -2*2*np.pi
gahigh = 2*np.pi

*tips*: We advise you to not put negative bounds, it will solicitate the neural network more than necessary for worse results in the end.
NB: the first angle is unimportant due to the axial symetry of the torus.

**Noise configurations, "none", "expo", "normal", "uniform" and the parameters (mean, std)**
mode = "normal"
first_p = 0
second_p = 0.2

NB: *std* is not squarred after.

**Noise amplification**
ampl = 0.1

NB: It is very sensitive, so be careful

**Save plots of generated tori (False pour not to, recommended if there is more than 150 datasets generated)**
show = False
save_fig = False

You need to show the fig to save it.
**Number of points (!! squarred after) for the dataset**
Ech = 7

 It gives a torus of 49 points.
 
 ## Run the models
 To run the model from a .csv datafile you need to run *predict_param.py*.

Be careful with the models you choose. The model named **idee** is the less efficient model but the most general one.
the numebr after the underscore is the number of points used at the tori generation. It means you cannot use this model for .csv file that have less than this number.

Then : 
*idee_169.h5* is the general model using a least 169 points. If you have more than 169 points the model will truncate the dataset (randomly).


