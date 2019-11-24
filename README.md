# data-science-tools
Collection of various data science tools, methods and tricks that I've picked up over the years. :)

## TODO
Make Jupyter Notebook versions at some point, but for starters I can just copy/paste to where I need stuff.

# Random Forest Regression
My Random Forest Regressor example is from [Kaggle](https://www.kaggle.com/nsrose7224/random-forest-regressor-accuracy-0-91).  

## Notes
### Encode with Dummy Variables or not?
- Scores from Kaggle
- #[0.906911897099843, 0.912672044577893, 0.9140890108660101, 0.9144505056433614, 0.914262282328547, 0.9139433343769434, 0.9163186431820812, 0.9166086725378967, 0.9160518483934068, 0.9167618968316904, 0.9165995923831073, 0.9172078680613065, 0.9164884688957832, 0.9161736086985683, 0.9165563453771916, 0.916971869620248, 0.9164909684909622, 0.9169219783868117, 0.9165771167149543]

- Scores without dummy variable encoding
- #[0.9035434557721174, 0.911559395258506, 0.9116842411645403, 0.9137533295545505, 0.9145245791031484, 0.9140046860090325, 0.9147700440114759, 0.9156499566036366, 0.9156030346267331, 0.914918468907167, 0.914751144603725, 0.91524986942177, 0.9155485461437795, 0.9156745152151314, 0.9151416967009058, 0.9150332956454269, 0.9153819969209948, 0.9152587078577952, 0.9156019956278598]

*slightly worse, so even with RF Regression it **is** a good idea to encode variables to dummies (although the difference is marginal)*

-- scores with dummy variables but without scaling
- #[0.9073091037173004, 0.9117271973020955, 0.9134276558222074, 0.9149089218418027, 0.9146367394753295, 0.9167258249950438, 0.9151800535573515, 0.9162655888021767, 0.9165045474833977, 0.91607739466341, 0.9161604756415774, 0.9160704574077145, 0.9169646947491383, 0.9171442074438759, 0.9172018808756496, 0.917392225747372, 0.9168447733468765, 0.9170469690592035, 0.9170299413806312]

*negligible difference*

### Number of n_estimators?
Run the model repeatedly to get a feel for the number of estimators.  
It'll allow for finding both the initial rise as well as later improvements (allows for quickly trying some things out as well as seeing the overall performance boundary)

### Scaling?
In this case it didn't make much difference whether scaling was used or not, the difference was negligible (actually slightly in favour of the unscaled model).
Generally, it is not necessary to use scaling.  
Scaling the inputs helps to avoid the situation, when one or several features dominate others in magnitude, as a result, the model hardly picks up the contribution of the smaller scale variables, even if they are strong.  
But if the target is scaled, the mean squared error is automatically scaled.  
MSE>1 automatically means that the model is doing worse than a constant (naive) prediction.