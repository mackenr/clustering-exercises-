from wrangle_zillow import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor,ElasticNet,LassoLarsIC,BayesianRidge,ARDRegression 
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import GridSearchCV


def regmodelbest(X_train, y_train, X_validate, y_validate, X_test, y_test, random=123):
    '''
    in the future this should be be updated for more automation. 
    https://machinelearninghd.com/gridsearchcv-hyperparameter-tuning-sckit-learn-regression-classification/
    offers some examples
    Currently it allows for the best models to be visualized. I will then apply a gradient search to the best model
    to optimize further

    Started adding a dictionary to pick the best model but other models could be added
    
    
    
    '''

    ## Baseline


    # make a list to store model rmse in and dict to pull best model from
    rmse_list=[]
    model_name_list=[]
    model_name_not_base=[]
    model_list=[]

  

    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    ## adding y_test to be used at the end
    y_test = pd.DataFrame(y_test)

    # 1. Predict taxvaluedollarcnt_pred_mean
    taxvaluedollarcnt_pred_mean = y_train['taxvaluedollarcnt'].mean()
    y_train['taxvaluedollarcnt_pred_mean'] = taxvaluedollarcnt_pred_mean
    y_validate['taxvaluedollarcnt_pred_mean'] = taxvaluedollarcnt_pred_mean

    # 2. compute taxvaluedollarcnt_pred_median
    taxvaluedollarcnt_pred_median = y_train['taxvaluedollarcnt'].median()
    y_train['taxvaluedollarcnt_pred_median'] = taxvaluedollarcnt_pred_median
    y_validate['taxvaluedollarcnt_pred_median'] = taxvaluedollarcnt_pred_median

    # 3. RMSE of taxvaluedollarcnt_pred_mean
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_mean)**(1/2)

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE using Mean')

    # 4. RMSE of taxvaluedollarcnt_pred_median
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_median)**(1/2)
    # print(f'''RMSE using Median\nTrain/In-Sample: {rmse_train:.4g}
    #       \nValidate/Out-of-Sample: \n {rmse_validate:.4g}\n\n''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE using Median')







    # 6. Compute taxvaluedollarcnt_pred_mode
    taxvaluedollarcnt_pred_mode = y_train['taxvaluedollarcnt'].mode().values[0]
    y_train['taxvaluedollarcnt_pred_mode'] = taxvaluedollarcnt_pred_mode
    y_validate['taxvaluedollarcnt_pred_mode'] = taxvaluedollarcnt_pred_mode

    # 7. RMSE of taxvaluedollarcnt_pred_mode

    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_median)**(1/2)

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE using Mode')








    ## RMSE for OLM 

    # create the model object
    lm = LinearRegression(normalize=True)
  

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.taxvaluedollarcnt)
   


    # predict train
    y_train['taxvaluedollarcnt_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm)**(1/2)


    # print(f'''RMSE for OLM \nTrain/In-Sample: {rmse_train:.4g}
    #       \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for OLM ')
    # model_name_not_base.append('RMSE for OLM' )
    # model_list.append(lm)







    ## RMSE for Elastic Net

    # create the model object
    Enm = ElasticNet(tol=1e-6)
  

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    Enm.fit(X_train, y_train.taxvaluedollarcnt)
   


    # predict train
    y_train['taxvaluedollarcnt_pred_elm'] = Enm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_elm)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_elm'] = Enm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_elm)**(1/2)


    # print(f'''RMSE for OLM \nTrain/In-Sample: {rmse_train:.4g}
    #       \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for Elastic Net Model ')
    # model_name_not_base.append('RMSE for OLM' )
    # model_list.append(lm)





    

     ##  RMSE for LassoLarsIC


    # create the model object
    larIC = LassoLarsIC()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    larIC.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lars'] = larIC.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lars)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lars'] = larIC.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lars)**(1/2)


    # print(f'''RMSE for Lasso + Lars\nTraining/In-Sample: :  {rmse_train:.4g}
    #  \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for LassoLarsIC')

    # model_name_not_base.append('RMSE for LassoLarsIC ')
    # model_list.append(lars)







     ##  RMSE for   ARDRegression


    # create the model object
    ardReg =     ARDRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    ardReg.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_ARD'] = ardReg.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_ARD)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_ARD'] = ardReg.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_ARD)**(1/2)


   

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for ARDRegression')

   








    ##  RMSE for Lasso + Lars 


    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lars)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lars)**(1/2)


    # print(f'''RMSE for Lasso + Lars\nTraining/In-Sample: :  {rmse_train:.4g}
    #  \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for Lasso + Lars ')

    # model_name_not_base.append('RMSE for Lasso + Lars ')
    # model_list.append(lars)


   



    ## RMSE for BayesianRidge

    
    
    
    # create the model object
    bayRidge =  BayesianRidge(n_iter=int(1e4),tol=1e-8) 

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    bayRidge.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_bayRidge'] = bayRidge.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_bayRidge)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_bayRidge'] = bayRidge.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_bayRidge)**(1/2)


    # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
    #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('BayesianRidge ')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)











    ## RMSE for Tweedie, power=1 & alpha=0

    
    
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_glm)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_glm)**(1/2)


    # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
    #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)




    ## RMSE for Tweedie, power=0 & alpha=0

    
    
    
    # create the model object
    glm = TweedieRegressor(power=0, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_glm)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_glm)**(1/2)


    # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
    #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for Tweedie, power=0 & alpha=0')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)


      ## RMSE for Tweedie, power=(1,2) & alpha=0

    
    
    
    # create the model object
    glm = TweedieRegressor(power=1.5, alpha=0.25)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_glm)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_glm)**(1/2)


    # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
    #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for Tweedie, power=1.5 & alpha=0')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)








    ## Prep for the 2nd degree Polynomial model

    #  make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree2 = pf.transform(X_validate)

    ## RMSE for the 2nd Degree Polynomial Model 


    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm2)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm2)**(1/2)

    # print(f'''RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: :  {rmse_train:.4g}
    #  \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for 2nd Degree Polynomial Model')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)









     ## Prep for the degree3 Polynomial model

    #  make the polynomial features to get a new set of features
    pf3 = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf3.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree3 = pf3.transform(X_validate)


    ## RMSE for the 2nd Degree Polynomial Model 


    # create the model object
    lm3 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm3.fit(X_train_degree3, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm3'] = lm3.predict(X_train_degree3)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm3)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm3'] = lm3.predict(X_validate_degree3)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm3)**(1/2)

    # print(f'''RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: :  {rmse_train:.4g}
    #  \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree3 Polynomial Model')
    model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    model_list.append(glm)










     ## Prep for the degree4 Polynomial model

    #  make the polynomial features to get a new set of features
    pf4 = PolynomialFeatures(degree=4)

    # fit and transform X_train_scaled
    X_train_degree4 = pf4.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree4 = pf4.transform(X_validate)


    ## RMSE for the 2nd Degree Polynomial Model 


    # create the model object
    lm4 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm4.fit(X_train_degree4, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm4'] = lm4.predict(X_train_degree4)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm4)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm4'] = lm4.predict(X_validate_degree4)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm4)**(1/2)

    # print(f'''RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: :  {rmse_train:.4g}
    #  \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree4 Polynomial Model')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)




 ## Prep for the degree5 Polynomial model

    #  make the polynomial features to get a new set of features
    pf5 = PolynomialFeatures(degree=5)

    # fit and transform X_train_scaled
    X_train_degree5 = pf5.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree5 = pf5.transform(X_validate)


    ## RMSE for the 5th Degree Polynomial Model 


    # create the model object
    lm5 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm5.fit(X_train_degree5, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm5'] = lm5.predict(X_train_degree5)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm5)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm5'] = lm5.predict(X_validate_degree5)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm5)**(1/2)


    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree5 Polynomial Model')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)






     ## Prep for the degree6 Polynomial model
  
    #  make the polynomial features to get a new set of features
    pf6 = PolynomialFeatures(degree=6)

    # fit and transform X_train_scaled
    X_train_degree6 = pf6.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree6 = pf6.transform(X_validate)

    ## RMSE for the 6th Degree Polynomial Model 


    # create the model object
    lm6 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm6.fit(X_train_degree6, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm6'] = lm6.predict(X_train_degree6)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm6)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm6'] = lm6.predict(X_validate_degree6)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm6)**(1/2)


    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree6 Polynomial Model')
    model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    model_list.append(glm)



     ## Prep for the degree7 Polynomial model
  
    #  make the polynomial features to get a new set of features
    pf7 = PolynomialFeatures(degree=7)

    # fit and transform X_train_scaled
    X_train_degree7 = pf7.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree7 = pf7.transform(X_validate)


    ## RMSE for the 7th Degree Polynomial Model 


    # create the model object
    lm7 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm7.fit(X_train_degree7, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm7'] = lm7.predict(X_train_degree7)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm7)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm7'] = lm7.predict(X_validate_degree7)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm7)**(1/2)


    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree7 Polynomial Model')

     ## Prep for the degree8 Polynomial model
 
    #  make the polynomial features to get a new set of features
    pf8 = PolynomialFeatures(degree=8)

    # fit and transform X_train_scaled
    X_train_degree8 = pf8.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree8 = pf8.transform(X_validate)


    ## RMSE for the 8th Degree Polynomial Model 


    # create the model object
    lm8 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm8.fit(X_train_degree8, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm8'] = lm8.predict(X_train_degree8)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm8)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm8'] = lm8.predict(X_validate_degree8)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm8)**(1/2)


    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree8 Polynomial Model')
    model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    model_list.append(glm)







    modelsresmedict=dict(zip(model_name_list,rmse_list))
    rmseDF=pd.DataFrame(data=modelsresmedict,index=['Train','Validate'])
    rmseDF=rmseDF.T
    rmseDF['diff']=rmseDF.Train-rmseDF.Validate
    rmseDF['abs_diff']=rmseDF['diff'].apply(abs)
    rmseDF['abs_percent_change']=(rmseDF['abs_diff']/rmseDF.Train)*100
    
    rmseDF.sort_values(by=['abs_diff'],inplace=True)






    # plot to visualize actual vs predicted. 
   
 
    return rmseDF








def single_split_many_return(df,mvp,totarget='logerror'):
    '''
    
  
    returns fulllist,fullscalledlist,lalist,lascaled,oclist,ocscalledlist   
    where they all have their respective X_train, y_train, X_validate, y_validate, X_test, y_test in position 0...5 
    the nonscaled lists have their respective non split test for viz purposes in position 6
    
    
    '''
    fulllist=[]
    

    oclist=[]
    


    lalist=[]

    venturalist=[]
   

    train, validate, test=train_validate_test(df) 
     # make, fit, use:
     ##LA
    la_train=train[train.county=='LA']
    la_validate=validate[validate.county=='LA']
    la_test=test[test.county=='LA']

    ##Orange
    orange_train=train[train.county=='Orange']
    orange_validate=validate[validate.county=='Orange']
    orange_test=test[test.county=='Orange']


    ##Ventura
    ventura_train=train[train.county=='Ventura']
    ventura_validate=validate[validate.county=='Ventura']
    ventura_test=test[test.county=='Ventura']


     




    X_train, y_train, X_validate, y_validate, X_test, y_test=bigX_little_y(train,  validate,  test, target=totarget)


    




   


    
   



     
   

    X_la_train, y_la_train, X_la_validate, y_la_validate, X_la_test, y_la_test=bigX_little_y(la_train,  la_validate,  la_test, target=totarget)
    

   
   




    
    



    X_orange_train, y_orange_train, X_orange_validate, y_orange_validate, X_orange_test, y_orange_test=bigX_little_y(orange_train,  orange_validate,  orange_test, target=totarget)




    X_ventura_train, y_ventura_train, X_ventura_validate, y_ventura_validate, X_ventura_test, y_ventura_test=bigX_little_y(ventura_train,  ventura_validate,  ventura_test, target=totarget)

    fulllist.append(X_train)
    fulllist.append(y_train)
    fulllist.append(X_validate)
    fulllist.append(y_validate)
    fulllist.append(X_test)
    fulllist.append(y_test)
    fulllist.append(train)


    lalist.append(X_la_train)
    lalist.append(y_la_train)
    lalist.append(X_la_validate)
    lalist.append(y_la_validate)
    lalist.append(X_la_test)
    lalist.append(y_la_test)
    lalist.append(la_train)

    oclist.append(X_orange_train)
    oclist.append(y_orange_train)
    oclist.append(X_orange_validate)
    oclist.append(y_orange_validate)
    oclist.append(X_orange_test)
    oclist.append(y_orange_test)
    oclist.append(orange_train)






    venturalist.append(X_ventura_train)
    venturalist.append(y_ventura_train)
    venturalist.append(X_ventura_validate)
    venturalist.append(y_ventura_validate)
    venturalist.append(X_ventura_test)
    venturalist.append(y_ventura_test)
    venturalist.append(ventura_train)

    train=train.drop(columns=['transaction_month', 'yearbuilt','longitude','latitude'])
    validate=validate.drop(columns=['transaction_month','yearbuilt','longitude','latitude'])
    test=test.drop(columns=['transaction_month','yearbuilt','longitude','latitude'])
    



    train_scaled=train
    validate_scaled=validate
    test_scaled=test
    
   


    la_train_scaled=train_scaled[train_scaled.county=='LA']
    la_validate_scaled=validate_scaled[validate_scaled.county=='LA']
    la_test_scaled=test_scaled[test_scaled.county=='LA']

    orange_train_scaled=train_scaled[train_scaled.county=='Orange']
    orange_validate_scaled=validate_scaled[validate_scaled.county=='Orange']
    orange_test_scaled=test_scaled[test_scaled.county=='Orange']

    ventura_train_scaled=train_scaled[train_scaled.county=='Ventura']
    ventura_validate_scaled=validate_scaled[validate_scaled.county=='Ventura']
    ventura_test_scaled=test_scaled[test_scaled.county=='Ventura']



     #reasoning location is accounted for by the lat long feature



    la_train_scaled=la_train_scaled.drop(columns=['county','fips'   ])
    la_validate_scaled=la_validate_scaled.drop(columns=['county','fips'   ])
    la_test_scaled=la_test_scaled.drop(columns=['county','fips'   ])

    orange_train_scaled=orange_train_scaled.drop(columns=['county','fips'   ])
    orange_validate_scaled=orange_validate_scaled.drop(columns=['county','fips'   ])
    orange_test_scaled=orange_test_scaled.drop(columns=['county','fips'   ])


    ventura_train_scaled=ventura_train_scaled.drop(columns=['county','fips'   ])
    ventura_validate_scaled=ventura_validate_scaled.drop(columns=['county','fips'   ])
    ventura_test_scaled=ventura_test_scaled.drop(columns=['county','fips'   ])


    

   
  
    train_scaled=train_scaled.drop(columns=['county','fips'   ])
    validate_scaled=validate_scaled.drop(columns=['county','fips'   ])
    test_scaled=test_scaled.drop(columns=['county','fips'   ])



    # make the object, put it into the variable scaler
    scaler = MinMaxScaler()
    # fit the object to my data:
    

    






    X_scale_train, y_scale_train, X_scale_validate, y_scale_validate, X_scale_test, y_scale_test=bigX_little_y(train_scaled,  validate_scaled,  test_scaled, target=totarget)

    
    X_la_train_scaled, y_la_train_scaled, X_la_validate_scaled, y_la_validate_scaled, X_la_test_scaled, y_la_test_scaled=bigX_little_y(la_train_scaled,  la_validate_scaled,  la_test_scaled, target=totarget)  

    X_orange_train_scaled, y_orange_train_scaled, X_orange_validate_scaled, y_orange_validate_scaled, X_orange_test_scaled, y_orange_test_scaled=bigX_little_y(orange_train_scaled,  orange_validate_scaled,  orange_test_scaled, target=totarget)

    X_ventura_train_scaled, y_ventura_train_scaled, X_ventura_validate_scaled, y_ventura_validate_scaled, X_ventura_test_scaled, y_ventura_test_scaled=bigX_little_y(ventura_train_scaled,  ventura_validate_scaled,  ventura_test_scaled, target=totarget)

    # mvp=list(set(mvp)-{'taxvaluedollarcnt'})


    # X_train,X_validate,X_test,X_la_train,X_la_validate,X_la_test,X_orange_train,X_orange_validate,X_orange_test=mvpModels(X_train,X_validate,X_test,X_la_train,X_la_validate,X_la_test,X_orange_train,X_orange_validate,X_orange_test,mvp)

    # X_scale_train,X_scale_validate,X_scale_test,X_la_train_scaled, X_la_validate_scaled,  X_la_test_scaled, X_orange_train_scaled,  X_orange_validate_scaled,  X_orange_test_scaled=mvpModels(X_scale_train,X_scale_validate,X_scale_test,X_la_train_scaled, X_la_validate_scaled,  X_la_test_scaled, X_orange_train_scaled,  X_orange_validate_scaled,  X_orange_test_scaled, mvp)



     

    # orange_train_scaled=encodeforZillow(orange_train_scaled)  

    # orange_validate_scaled=encodeforZillow(orange_validate_scaled)        

    # orange_test_scaled=encodeforZillow(orange_test_scaled)
    #train_scaled= scaler.fit_transform(train_scaled)
    X_scale_train=scaler.fit_transform(X_scale_train)
    X_scale_validate= scaler.transform(X_scale_validate) 
    X_scale_test= scaler.transform(X_scale_test)

    X_la_train_scaled=scaler.fit_transform(X_la_train_scaled)
    X_la_validate_scaled=scaler.transform(X_la_validate_scaled)
    X_la_test_scaled=scaler.transform(X_la_test_scaled)

    X_orange_validate_scaled=scaler.fit_transform(X_orange_validate_scaled)
    X_orange_validate_scaled=scaler.transform(X_orange_validate_scaled)
    X_orange_test_scaled=scaler.transform(X_orange_test_scaled)

    X_ventura_validate_scaled=scaler.fit_transform(X_ventura_validate_scaled)
    X_ventura_validate_scaled=scaler.transform(X_ventura_validate_scaled)
    X_ventura_test_scaled=scaler.transform(X_ventura_test_scaled)
   
   

    





    fullscalledlist=[]

    ocscalledlist=[]
    lascaled=[]
    venturascalledlist=[]
   
    
    
    

    fullscalledlist.append(X_scale_train)
    fullscalledlist.append(y_scale_train)
    fullscalledlist.append(X_scale_validate)
    fullscalledlist.append(y_scale_validate)
    fullscalledlist.append(X_scale_test)
    fullscalledlist.append(y_scale_test)

    

    lascaled.append(X_la_train_scaled)
    lascaled.append(y_la_train_scaled)
    lascaled.append(X_la_validate_scaled)
    lascaled.append(y_la_validate_scaled)
    lascaled.append(X_la_test_scaled)
    lascaled.append(y_la_test_scaled)

   

    ocscalledlist.append(X_orange_train_scaled)
    ocscalledlist.append(y_orange_train_scaled)
    ocscalledlist.append(X_orange_validate_scaled)
    ocscalledlist.append(y_orange_validate_scaled)
    ocscalledlist.append(X_orange_test_scaled)
    ocscalledlist.append(y_orange_test_scaled)



    venturascalledlist.append(X_ventura_train_scaled)
    venturascalledlist.append(y_ventura_train_scaled)
    venturascalledlist.append(X_ventura_validate_scaled)
    venturascalledlist.append(y_ventura_validate_scaled)
    venturascalledlist.append(X_ventura_test_scaled)
    venturascalledlist.append(y_ventura_test_scaled)












    # train=pd.merge(X_train, y_train,left_index=True,right_index=True)

    # train_la=pd.merge(X_la_train, y_la_train,left_index=True,right_index=True)

    # orange_train=pd.merge(X_orange_train, y_orange_train,left_index=True,right_index=True)

    # orange_train=pd.merge(X_orange_train, y_orange_train,left_index=True,right_index=True)


    return   fulllist,fullscalledlist,lalist,lascaled,oclist,ocscalledlist,venturalist,venturascalledlist
  



def goldenvisvars(m=1.75):
    '''
    m is our scaler
    created height and width ratios and aspect based on the golden ratio 
    
    
    '''
    h=21**(1/2)
 
    w=21**(1/2)+13**(1/2)
    h=h*m
    w=w*m  
    aspect=(1+(5)**.5)/2
    h_int=int(h*.95)

    return h,w,aspect,h_int





def mvpModels(X_train,X_validate,X_test,X_la_train,X_la_validate,X_la_test,X_orange_train,X_orange_validate,X_orange_test,mvp):
    '''
    
    reduces the restective big X dataframes to the mvp and scales them
    
    
    '''
    X_train=X_train[mvp]
    X_validate=X_validate[mvp]
    X_test=X_test[mvp]
    X_la_train=X_la_train[mvp]
    X_la_validate=X_la_validate[mvp]
    X_la_test=X_la_test[mvp]
    X_orange_train=X_orange_train[mvp]
    X_orange_validate=X_orange_validate[mvp]
    X_orange_test=X_orange_test[mvp]
   




    return X_train,X_validate,X_test,X_la_train,X_la_validate,X_la_test,X_orange_train,X_orange_validate,X_orange_test





def fullTest(X_train, y_train, X_validate, y_validate, X_test, y_test):
    '''
    this is created simply to minimize the code in the main juoyternotebook 
    
    '''
    
    
     # create the model object
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    ## adding y_test to be used at the end
    y_test = pd.DataFrame(y_test)


    rmse_list=[]
    model_name_list=[]


    

     ## Prep for the degree3 Polynomial model

    #  make the polynomial features to get a new set of features
    pf3 = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf3.fit_transform(X_train)

    # transform X_validate_scaled 
    X_validate_degree3 = pf3.transform(X_validate)


     # transform X_validate_scaled 
    X_test_degree3 = pf3.transform(X_test)



    ## RMSE for the 2nd Degree Polynomial Model 


    # create the model object
    lm3 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm3.fit(X_train_degree3, y_train.taxvaluedollarcnt)

    # predict train
    y_train['taxvaluedollarcnt_pred_lm3'] = lm3.predict(X_train_degree3)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_lm3)**(1/2)

    # predict validate
    y_validate['taxvaluedollarcnt_pred_lm3'] = lm3.predict(X_validate_degree3)
    

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_lm3)**(1/2)

    # print(f'''RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: :  {rmse_train:.4g}
    #  \nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')

    rmse_list.append([rmse_train,rmse_validate])
    model_name_list.append('RMSE for degree3 Polynomial Model')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)





    # evaluate: rmse
    rmse_test= mean_squared_error(y_test.taxvaluedollarcnt, y_test.taxvaluedollarcnt_pred_ElasticNet)**(1/2)
  
    rmse_list.append([rmse_train,rmse_validate,rmse_test])
    # model_name_list.append('Elastic Net ')
    #
    modelsresmedict=dict(zip(model_name_list,rmse_list))
    rmseDF=pd.DataFrame(data=modelsresmedict,index=['Train','Validate','Test'])
    rmseDF=rmseDF.T
    rmseDF['Train_to_Val_diff']=rmseDF.Train-rmseDF.Validate
    rmseDF['Train_to_Val_abs_diff']=rmseDF['Train_to_Val_diff'].apply(abs)
    rmseDF['Train_to_Test_diff']=rmseDF.Train-rmseDF.Test
    rmseDF['Train_to_Test_abs_diff']=rmseDF['Train_to_Test_diff'].apply(abs)

     # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.scatter(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_ElasticNet,
            alpha=.5, color="yellow", s=100, label="Model: Elastic Net Train")
    plt.scatter(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_ElasticNet, 
                 alpha=.5, color="green", s=100, label="Model: Elastic Net Val")
    plt.scatter(y_test.taxvaluedollarcnt, y_test.taxvaluedollarcnt_pred_ElasticNet,
            alpha=.5, color="red", s=100, label="Model: Elastic Net Test")
    plt.legend()
    plt.xlabel("Actual Taxval")
    plt.ylabel("Predicted Actual Taxval")
    plt.title("Best with Full Dataset")
#      plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
#      plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    x=plt.show()


    return rmseDF,x



def laTestregmodelbest(X_la_train, y_la_train, X_la_validate, y_la_validate, X_la_test, y_la_test):
  '''
  this is created simply to minimize the code in the main juoyternotebook 
    
  '''




  # create the model object
  glm = TweedieRegressor(power=1.5, alpha=0.25)  
  


  # create the model object
  # We need y_la_train and y_la_validate to be dataframes to append the new columns with predicted values. 
  y_la_train = pd.DataFrame(y_la_train)
  y_la_validate = pd.DataFrame(y_la_validate)
  ## adding y_la_test to be used at the end
  y_la_test = pd.DataFrame(y_la_test)


  rmse_list=[]
  model_name_list=[]
  # fit the model to our training data. We must specify the column in y_la_train, 
  # since we have converted it to a dataframe from a series! 
  glm.fit(X_la_train, y_la_train)
  # predict train
  y_la_train['taxvaluedollarcnt_pred_Tweedie_pow_1andhalf'] = glm.predict(X_la_train)
  # evaluate: rmse
  rmse_train = mean_squared_error(y_la_train.taxvaluedollarcnt, y_la_train.taxvaluedollarcnt_pred_Tweedie_pow_1andhalf)**(1/2)
  # predict validate
  y_la_validate['taxvaluedollarcnt_pred_Tweedie_pow_1andhalf'] = glm.predict(X_la_validate)
  # evaluate: rmse
  rmse_validate = mean_squared_error(y_la_validate.taxvaluedollarcnt, y_la_validate.taxvaluedollarcnt_pred_Tweedie_pow_1andhalf)**(1/2)
  # predict test
  y_la_test['taxvaluedollarcnt_pred_Tweedie_pow_1andhalf'] = glm.predict(X_la_test)
  # evaluate: rmse
  rmse_test= mean_squared_error(y_la_test.taxvaluedollarcnt, y_la_test.taxvaluedollarcnt_pred_Tweedie_pow_1andhalf)**(1/2)
  # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
  #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')
  rmse_list.append([rmse_train,rmse_validate,rmse_test])
  model_name_list.append('TweedieRegressor (power=1.5, alpha=0.25)')
  # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
  # model_list.append(glm)

  modelsresmedict=dict(zip(model_name_list,rmse_list))
  rmseDF=pd.DataFrame(data=modelsresmedict,index=['Train','Validate','Test'])
  rmseDF=rmseDF.T
  rmseDF['Train_to_Val_diff']=rmseDF.Train-rmseDF.Validate
  rmseDF['Train_to_Val_abs_diff']=rmseDF['Train_to_Val_diff'].apply(abs)
  rmseDF['Train_to_Test_diff']=rmseDF.Train-rmseDF.Test
  rmseDF['Train_to_Test_abs_diff']=rmseDF['Train_to_Test_diff'].apply(abs)


#   plt.figure(figsize=(16,8))
#   plt.plot(y_la_train.taxvaluedollarcnt,y_la_train.taxvaluedollarcnt, alpha=.5, color="blue", label='_nolegend_')
#   plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)
#   plt.scatter(y_la_train.taxvaluedollarcnt, y_la_train.taxvaluedollarcnt_pred_Tweedie_pow_1andhalf,
#           alpha=.5, color="yellow", s=100, label="Train: 'TweedieRegressor (power=1.5, alpha=0.25)'")
#   plt.scatter(y_la_validate.taxvaluedollarcnt, y_la_validate.taxvaluedollarcnt_pred_Tweedie_pow_1andhalf, 
#                alpha=.5, color="green", s=100, label="Validate: 'TweedieRegressor (power=1.5, alpha=0.25)'")
#   plt.scatter(y_la_test.taxvaluedollarcnt, y_la_test.taxvaluedollarcnt_pred_Tweedie_pow_1andhalf,
#           alpha=.5, color="red", s=100, label="Test TweedieRegressor (power=1.5, alpha=0.25)")
#   plt.legend()
#   plt.xlabel("Actual Taxval")
#   plt.ylabel("Predicted  Taxval")
#   plt.title("Best with LA Dataset")


  return rmseDF




def orangeTest(X_orange_train, y_orange_train, X_orange_validate, y_orange_validate, X_orange_test, y_orange_test):
    '''
    this is created simply to minimize the code in the main juoyternotebook 
    
    '''
    # create the model object
    # We need y_orange_train and y_orange_validate to be dataframes to append the new columns with predicted values. 
    y_orange_train = pd.DataFrame(y_orange_train)
    y_orange_validate = pd.DataFrame(y_orange_validate)
    ## adding y_orange_test to be used at the end
    y_orange_test = pd.DataFrame(y_orange_test)
    
    # create the model object
    Enm = ElasticNet(tol=1e-6)
    
        
    rmse_list=[]
    model_name_list=[]
    # fit the model to our training data. We must specify the column in y_orange_train, 
    # since we have converted it to a dataframe from a series! 
    Enm.fit(X_orange_train, y_orange_train.taxvaluedollarcnt)
    # predict train
    y_orange_train['taxvaluedollarcnt_pred_ENM'] = Enm.predict(X_orange_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_orange_train.taxvaluedollarcnt, y_orange_train.taxvaluedollarcnt_pred_ENM)**(1/2)
    # predict validate
    y_orange_validate['taxvaluedollarcnt_pred_ENM'] = Enm.predict(X_orange_validate)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_orange_validate.taxvaluedollarcnt, y_orange_validate.taxvaluedollarcnt_pred_ENM)**(1/2)
    # predict test
    y_orange_test['taxvaluedollarcnt_pred_ENM'] = Enm.predict(X_orange_test)
    # evaluate: rmse
    rmse_test= mean_squared_error(y_orange_test.taxvaluedollarcnt, y_orange_test.taxvaluedollarcnt_pred_ENM)**(1/2)
    # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
    #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')
    rmse_list.append([rmse_train,rmse_validate,rmse_test])
    model_name_list.append('RMSE for Elastic Net Model ')
    # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
    # model_list.append(glm)
    
    modelsresmedict=dict(zip(model_name_list,rmse_list))
    rmseDF=pd.DataFrame(data=modelsresmedict,index=['Train','Validate','Test'])
    rmseDF=rmseDF.T
    rmseDF['Train_to_Val_diff']=rmseDF.Train-rmseDF.Validate
    rmseDF['Train_to_Val_abs_diff']=rmseDF['Train_to_Val_diff'].apply(abs)
    rmseDF['Train_to_Test_diff']=rmseDF.Train-rmseDF.Test
    rmseDF['Train_to_Test_abs_diff']=rmseDF['Train_to_Test_diff'].apply(abs)


    # plt.figure(figsize=(16,8))
    # plt.scatter(y_orange_train.taxvaluedollarcnt, y_orange_train.taxvaluedollarcnt_pred_ENM,
    #         alpha=.5, color="yellow", s=100, label="Train: 'Elastic Net Model")
    # plt.scatter(y_orange_validate.taxvaluedollarcnt, y_orange_validate.taxvaluedollarcnt_pred_ENM, 
    #              alpha=.5, color="green", s=100, label="Validate: Elastic Net Model")
    # plt.scatter(y_orange_test.taxvaluedollarcnt, y_orange_test.taxvaluedollarcnt_pred_ENM,
    #         alpha=.5, color="red", s=100, label="Test Elastic Net Model")
    # plt.legend()
    # plt.xlabel("Actual Taxval")
    # plt.ylabel("Predicted Actual Taxval")
    # plt.title("Best with OC Dataset")
    




    return rmseDF









def encodeforZillow(df):
    notencodelist=['area',
        'taxvaluedollarcnt',
        'yearbuilt',
        'parcelid',
        'lotsizesquarefeet',
        'longitude',
        'latitude',
        'transactiondate_in_days',
        'logerror',
        'lotsize/area',
        'bedbath_harmean',
        'bedbath_harmeandividesarea',
        'sqrt(bed^2+bath^2)',
        'sqrt(bed^2+bath^2)dividesarea',
        'bathplusbathdividesarea',
        'sqrt(bed^2+bath^2)divides(lotsize/area)',
        'bedbath_harmean)divides(lotsize/area)',
        'latscaled',
        'longscaled',
        'latlongPythagC',
        'latlongPythagC*(sqrt(bed^2+bath^2)divides(lotsize/area))',
        'lotsizesquarefeet_wo_outliers']
    encodelist=list(set(df.columns)-set(notencodelist))

    df=pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=encodelist, sparse=False, drop_first=False, dtype=bool)
    return df