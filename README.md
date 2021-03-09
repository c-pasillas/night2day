# night2day
this is the start of my git project to help colloboration on my dissertaiton code

to do list 

config file to set paths but to be reachable whether i run full code or just one processing step at a time
call the config file in each processing step if necessary

adjust processonesample become 2 fxns "processCDF" and processarray" to more readily adjust if there are missing files and then make a processonesample that runs both if i am starting from scratch.

still need to look into xarray to see if i can "processonesample" and "processonecase" without saving crazy amounts of . nc files

POSTPROCESSING
- need to hae a code to rebuild the larger array ( for each time step) from the patches since flatten will not work to maintain spatial and temporal considerations.  This way I can have single image of the area for test cases.  Later I would be runnign the learned model on different data set and can hopefully reprocess image by time and area as well.
