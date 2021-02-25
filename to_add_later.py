#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#to_add_later

Created on Mon Jan 11 14:15:04 2021

@author: cpasilla
"""
#edited 11 Jan 2020 = all code relaitong to SWIR, HNCC ERFDNB in here

#this file will have the SWIR and HNCC/ERF codes/processes 
#for if we get to that point later. Will have to be adjusted 
#rather than comment out in the good code i may need it so its here
#not sure if I can make these arrays and upload and combine at end
#with other data or redo all the work


##################################################################
############ MAKING RAW ARRAY FOR A SINGLE TIMESTAMP##############
##################################################################
# PHD_proj_SINGLE_TIMESTEP.py
#STEP 1a: "colocate"  
#STEP 1b: "combine single channel arrays"


#load the channels and composites to the "scene"
original_scn.load(['M08', 'M09','M10','M11','dynamic_dnb','hncc_dnb','dnb_latitude','dnb_longitude','m_latitude','m_longitude'])

original_scn['M08].attrs['area']
original_scn['M09'].attrs['area']
original_scn['M10'].attrs['area']
original_scn['M11'].attrs['area']
original_scn['dynamic_dnb'].attrs['area']
original_scn['hncc_dnb'].attrs['area']



#print/confirming shape of the original files before data manipulation 
print('these are the start shapes of data files')
print(original_scn['M08'].shape)
print(original_scn['M09'].shape)
print(original_scn['M10'].shape)
print(original_scn['M11'].shape)
print(original_scn['dynamic_dnb'].shape)
print(original_scn['hncc_dnb'].shape)
print(original_scn['dnb_latitude'].shape)
print(original_scn['dnb_longitude'].shape)


print('this is the end or inital shape confirmation')


#now resample the Mband to the DNB with a new "Scene" role
print('starting resampling of Mbands')
Mband_resample_scn = original_scn.resample(resampler = 'nearest')
Mband_resample_scn['DNB'].shape == Mband_resample_scn['M08'].shape
Mband_resample_scn['M08'].attrs['area']

print('these are the colocated Mband shapes')
print(Mband_resample_scn['M08'].shape)
print(Mband_resample_scn['M09'].shape)
print(Mband_resample_scn['M10'].shape)
print(Mband_resample_scn['M11'].shape)
print(Mband_resample_scn['dynamic_dnb'].shape)
print(Mband_resample_scn['hncc_dnb'].shape)


#save the resampled Mbands as .nc files and images
############

Mband_resample_scn.save_dataset('M08', spath + DTG +'_RAW_COLOCATED_M08.png', writer='simple_image') #engine='netcdf4')
Mband_resample_scn.save_dataset('M08', spath + DTG +'_RAW_COLOCATED_M08.nc', writer='cf')
Mband_resample_scn.save_dataset('M09', spath + DTG +'_RAW_COLOCATED_M09.png', writer='simple_image') #engine='netcdf4')
Mband_resample_scn.save_dataset('M09', spath + DTG +'_RAW_COLOCATED_M09.nc', writer='cf')
Mband_resample_scn.save_dataset('M10', spath + DTG +'_RAW_COLOCATED_M10.png', writer='simple_image') #engine='netcdf4')
Mband_resample_scn.save_dataset('M10', spath + DTG +'_RAW_COLOCATED_M10.nc', writer='cf')
Mband_resample_scn.save_dataset('M11', spath + DTG +'_RAW_COLOCATED_M11.png', writer='simple_image') #engine='netcdf4')
Mband_resample_scn.save_dataset('M11', spath + DTG +'_RAW_COLOCATED_M11.nc', writer='cf')
Mband_resample_scn.save_dataset('dynamic_dnb', spath + DTG +'_RAW_COLOCATED_dynamic_dnb.png', writer='simple_image') #engine='netcdf4')
Mband_resample_scn.save_dataset('dynamic_dnb', spath + DTG +'_RAW_COLOCATED_dynamic_dnb.nc', writer='cf')
Mband_resample_scn.save_dataset('hncc_dnb', spath + DTG +'_RAW_COLOCATED_hncc_dnb.png', writer='simple_image') #engine='netcdf4')
Mband_resample_scn.save_dataset('hncc_dnb', spath + DTG +'_RAW_COLOCATED_hncc_dnb.nc', writer='cf')


# need to input M08-M11 here just like below

#hncc
k= Dataset( spath + DTG + "_COLOCATED_hncc_dnb.nc")
kk=k["hncc_dnb"][:].filled(-9999)
mk = kk[kk != -9999]
print("hncc_dnb max", numpy.max(kk), "hncc_dnb min", numpy.min(mk)  )

#dynamic
l= Dataset( spath + DTG + "_COLOCATED_dynamic_dnb.nc")
ll=l["dynamic_dnb"][:].filled(-9999)
ml = ll[ll != -9999]
print("dynamic_dnb max", numpy.max(ll), "dynamic_dnb min", numpy.min(ml)  )


#make the combined array
timestepconsolidated= numpy.stack((lat,long,aa,bb,cc,dd,ee,ff)) #replace with the M08-M11 an other dataset arrays
print(timestepconsolidated.shape)

[8] hncc, [9] dynamic


TA.createDimension('channel', 10) # variables loaded
#M12-M16= 5 #DNB , hncc, dynamic = 3 #lat/long =2

#build the variables/preallocate NetCDF variables for data storage.
# use createVariable method

M08_out=TA.createVariable('M08','f8', ('y','x'))      
M09_out=TA.createVariable('M09','f8', ('y','x'))
M10_out=TA.createVariable('M10','f8', ('y','x'))
M11_out=TA.createVariable('M11','f8', ('y','x'))

HNCC=TA.createVariable('HNCC','f8', ('y','x'))
ERFDNB=TA.createVariable('ERFDNB','f8', ('y','x'))

#passing data into the variables # adjust the position

M08_out[:]= timestepconsolidated[2,:,:]
M09_out[:]= timestepconsolidated[3,:,:]
M10_out[:]= timestepconsolidated[4,:,:]
M11_out[:]= timestepconsolidated[5,:,:]

HNCC[:]=timestepconsolidated[8,:,:]
ERFDNB[:]=timestepconsolidated[9,:,:]



#adding attributes
#adjust this for all M08-M11

M12_out.long_name = "M12" ;
M12_out.standard_name = "toa_brightness_temperature" ;
M12_out.units = "K" ;
M12_out.wavelength = "3.61, 3.7, 3.79"
M12_out.coordinates = "latitude longitude" ;

#
HNCC.long_name = "hncc_dnb" ;
HNCC.standard_name = "equalized_radiance" ;
HNCC.units = "1" ;
HNCC.wavelength = "pseudo albedo" ;
HNCC.coordinates = "latitude longitude" ;

ERFDNB.long_name = "ERFDNB" ;
ERFDNB.standard_name = "equalized_radiance" ;
ERFDNB.units = "1" ;
ERFDNB.wavelength = "TDB" ;
ERFDNB.coordinates = "latitude longitude" ;


##################################################################
################ MAKING A RAW CASE INITAL ARRAY###################
##################################################################
#PHD_proj_RAW_CASE_INITIAL_ARRAY
#STEP 2: combine all time steps to a single 4D array


#position we have saved them in above, adjust the DNBs as well
M08= 
M09=
M10=
M11=
DNB_HNCC=master[:,:,:,8],\
DNB_ZINKE=master[:,:,:,9],)




##################################################################
############# ARRAY SIZE REDUCTION TO USEFUL DATA SET#############
##################################################################
#"PHD_proj_ARRAY_REDUCTION_TO_VALIDAREA
#STEP 3a: remove the -9999 columns 
#STEP 3b: further reduce to an array divisible by 256x256 ( make sure it matches size and columns from the first data set ( ie removes same columns not just same #, its possible this one is smaller and if so then have adjust the original as well)


##################################################################
############# NOW DO MATH MANIPULATIONS FOR ML PREP###############
##################################################################
#PHD_proj_MATH_MANIPULATION
#STEP 4: normalize the SWIR and enhancements

gg = Dataset(spath + DTG + "_VALIDAREA_M08.nc") #M08
hh = = Dataset(spath + DTG + "_VALIDAREA_M09.nc") #M09
ii = Dataset(spath + DTG + "_VALIDAREA_M10.nc") #M11
jj = Dataset(spath + DTG + "_VALIDAREA_M11.nc") #DNBM11

kk = Dataset("/Users/cpasilla/PHD_WORK/RAW_DATA/2020/JAN/VALIDAREA_hncc_dnb.nc")
ll = Dataset("/Users/cpasilla/PHD_WORK/RAW_DATA/2020/JAN/VALIDAREA_dynamic_dnb.nc")

M08 = gg['M08'][:]
M09 = hh['M09'][:]
M10 = ii['M10'][:]
M11 = jj['M11'][:]
HNCC = kk['hncc_dnb'][:]
ERFDNB = ll['dynamic_dnb'][:]


print("M08 raw min/max values are", np.amax(M08),"and", np.amin (M08))
print("M09 raw min/max values are", np.amax(M09),"and", np.amin (M09))
print("M10 raw min/max values are", np.amax(M10),"and", np.amin (M10))
print("M11 raw min/max values are", np.amax(M11),"and", np.amin (M11))
print("HNCC raw min/max values are", np.amax(HNCC),"and", np.amin (HNCC))
print("ERFDNB raw min/max values are", np.amax(ERFDNB),"and", np.amin (ERFDNB))


#reflective bands
    #M11    0.12 - 31.8
    #M10    1.2 - 71.2
    #M09    0.6 - 77.1
    #M08    3.5 - 164.9


#M11mn = 0.12
#M11mx = 31.8
#M10mn = 1.2
#M10mx = 71.2
#M09mn = 0.6
#M09mx = 77.1
#M08mn = 3.5M08mx = 164.9
DNBmx = 3E-7
DNBmn = 2E-10
#HNCCmx
#HNCCmn
#ERFDNBmx
#ERFDNBmn


#DNB AND  DNB composites
#norm values for then HNCC and ERFDNB go here

# SWIR

#switch formula because radiances
# (Value - min)/(max-min) for channel/bands in visible SWIR
M11norm = (M11[:]-M11mn)/(M11mx-M11mn)
print('M11norm max min are', np.max(M11norm), np.min(M11norm))
M11norm[M11norm>1]=1
print('M11norm  final max min are', np.max(M11norm), np.min(M11norm))

M10norm = (M10[:]-M10mn)/(M10mx-M10mn)
print('M10norm max min are', np.max(M10norm), np.min(M10norm))
M10norm[M10norm>1]=1
print('M10norm  final max min are', np.max(M10norm), np.min(M10norm))

M09norm = (M09[:]-M09mn)/(M09mx-M09mn)
print('M09norm max min are', np.max(M09norm), np.min(M09norm))
M09norm[M09norm>1]=1
print('M09norm  final max min are', np.max(M09norm), np.min(M09norm))

M08norm = (M08[:]-M08mn)/(M08mx-M08mn)
print('M08norm max min are', np.max(M08norm), np.min(M08norm))
M08norm[M08norm>1]=1
print('M08norm  final max min are', np.max(M08norm), np.min(M08norm))

 #create dimensions
# other DNB enhancements and norm = 6


#createvariables

#HNCC=TA.createVariable('HNCC','f8', ('y','x'))
#HNCCnorm=TA.createVariable('HNCCnorm','f8', ('y','x'))
#ERFDNB=TA.createVariable('ERFDNB','f8', ('y','x'))
#ERFDNBnorm=TA.createVariable('ERFDNBnorm','f8', ('y','x'))

# SWIR channels M08-M11
M11_out=TA.createVariable('M11','f8', ('y','x'))
M10_out=TA.createVariable('M10','f8', ('y','x'))
M09_out=TA.createVariable('M09','f8', ('y','x'))
M08_out=TA.createVariable('M08','f8', ('y','x'))

M11norm_out=TA.createVariable('M11norm','f8', ('y','x'))
M10norm_out=TA.createVariable('M10norm','f8', ('y','x'))
M09norm_out=TA.createVariable('M09norm','f8', ('y','x'))
M08norm_out=TA.createVariable('M08norm','f8', ('y','x'))


#passing data into the variables

#other enhancements
#HNCC[:]=HNCC
#HNCCnorm[:]=HNCCnorm
#ERFDNB[:]=ERFDNB
#ERFDNBnorm[:]=ERFDNBnorm

# SWIR M08-M11
#M11_out[:]= M11
#M10_out[:]= M10
#M09_out[:]= M09
#M08_out[:]= M08

#M11norm_out[:]= M11norm
#M10norm_out[:]= M10norm
#M09norm_out[:]= M09norm
#M08norm_out[:]= M08norm


#adding attributes

#other DNB composites


#HNCC.long_name = "hncc_dnb" ;
#HNCC.standard_name = "equalized_radiance" ;
#HNCC.units = "1" ;
#HNCC.wavelength = "pseudo albedo" ;
#HNCC.coordinates = "latitude longitude" ;

#ERFDNB.long_name = "ERFDNB" ;
#ERFDNB.standard_name = "equalized_radiance" ;
#ERFDNB.units = "1" ;
#ERFDNB.wavelength = "TDB" ;
#ERFDNB.coordinates = "latitude longitude" ;

#HNCCnorm.long_name = "hnccnorm_dnb" ;
#HNCCnorm.standard_name = "normalzied hncc albedos" ;
#HNCCnorm.units = "0-1" ;
#HNCCnorm.coordinates = "latitude longitude" ;

#ERFDNBnorm.long_name = "ERFDNB normalized" ;
#ERFDNBnorm.standard_name = "normalized ERF radiance" ;
#ERFDNBnorm.units = "0-1" ;
#ERFDNBnorm.coordinates = "latitude longitude" ;


#SWIR M08-M11
M11_out.long_name = "M11" ;
M11_out.standard_name = "toa_bidirectional_reflectance" ;
M11_out.units = "%" ;
M11_out.wavelength = "2.225, 2.25, 2.275"
M11_out.coordinates = "latitude longitude" ;

M10_out.long_name = "M10" ;
M10_out.standard_name = "toa_bidirectional_reflectance" ;
M10_out.units = "%" ;
M10_out.wavelength = "1.58, 1.61, 1.64" 
M10_out.coordinates = "latitude longitude" ;

M09_out.long_name = "M09" ;
M09_out.standard_name = "toa_bidirectional_reflectance" ;
M09_out.units = "%" ;
M09_out.wavelength = "1.371, 1.378, 1.386" 
M09_out.coordinates = "latitude longitude" ;

M08_out.long_name = "M08" ;
M08_out.standard_name = "toa_bidirectional_reflectance" ;
M08_out.units = "%" ;
M08_out.wavelength = "1.23, 1.24, 1.25"
M08_out.coordinates = "latitude longitude" ;

M11norm_out.long_name = "M11norm" ;
M11norm_out.standard_name = "normalized toa_bidirectional_reflectance" ;
M11norm_out.units = "0-1" ;
M11norm_out.wavelength = "2.225, 2.25, 2.275"
M11norm_out.coordinates = "latitude longitude" ;

M10norm_out.long_name = "M10norm" ;
M10norm_out.standard_name = "normalized toa_bidirectional_reflectance" ;
M10norm_out.units = "0-1" ;
M10norm_out.wavelength = "1.58, 1.61, 1.64" 
M10norm_out.coordinates = "latitude longitude" ;

M09norm_out.long_name = "M09norm" ;
M09norm_out.standard_name = "normalized toa_bidirectional_reflectance" ;
M09norm_out.units = "0-1" ;
M09norm_out.wavelength = "1.371, 1.378, 1.386" 
M09norm_out.coordinates = "latitude longitude" ;

M08norm_out.long_name = "M08norm" ;
M08norm_out.standard_name = "normalized toa_bidirectional_reflectance" ;
M08norm_out.units = "0-1" ;
M08norm_out.wavelength = "1.23, 1.24, 1.25"
M08norm_out.coordinates = "latitude longitude" ;



print("M11 max is", np.max(M11), "M11 min is", np.min(M11))
Rma=M11[M11 != -9999]
Rma.max()
Rma.min()
A=(M11>0).sum()
N=(M11==-9999).sum()
D=N+A
print("RM11 max is", np.max(Rma), "RM11 min is", np.min(Rma))
print("M11 has this many valid", A,)
print("M11 has this many null values", N,)
print("M11 has this many  values", D,)
#########
########
print("M10 max is", np.max(M10), "M10 min is", np.min(M10))
Rma=M10[M10 != -9999]
Rma.max()
Rma.min()
A=(M10>0).sum()
N=(M10==-9999).sum()
D=N+A
print("RM10 max is", np.max(Rma), "RM10 min is", np.min(Rma))
print("M10 has this many valid", A,)
print("M10 has this many null values", N,)
print("M10 has this many  values", D,)
######3
print("M09 max is", np.max(M09), "M09 min is", np.min(M09))
Rma=M09[M09 == -9999]
Rma.max()
Rma.min()
A=(M09>0).sum()
N=(M09==-9999).sum()
D=N+A
print("RM09 max is", np.max(Rma), "RM09 min is", np.min(Rma))
print("M09 has this many valid", A,)
print("M09 has this many null values", N,)
print("M09 has this many  values", D,)
######
print("M08 max is", np.max(M08), "M08 min is", np.min(M08))
Rma=M08[M08 == -9999]
Rma.max()
Rma.min()
A=(M08>0).sum()
N=(M08==-9999).sum()
D=N+A
print("RM08 max is", np.max(Rma), "RM08 min is", np.min(Rma))
print("M08 has this many valid", A,)
print("M08 has this many null values", N,)
print("M08 has this many  values", D,)



#reset what positions is the DNB norm once make the master
DNBnorm = aa['normchannels[2]']
#HNCCnorm = aa['hncc_norm']
#ERFDNBnorm = aa["dynamic_norm']


##################################################################
################MAKE THE ARRAYS FOR ML INPUT######################
##################################################################
#STEP 5: Make the files needed (in/outputs predictors/ands) to go into a ML run
#PHD_proj_ML_PREP


spath =
DTG =
zz = Dataset(spath + DTG+ "_TIMESTEP_MASTER_ALLDATA.nc") #open total array

# lat and long

#lat = zz["Latitude"]
#long = zz["Longitude"]
#norm band values
B_M08norm = zz['Total_Data/M08norm'][:]
B_M09norm = zz['Total_Data/M09norm'][:]
B_M10norm = zz['Total_Data/M10norm'][:]
B_M11norm = zz['Total_Data/M111norm'][:]
B_HNCCnorm = zz['Total_Data/HNCCnormmoon'][:]
B_ERFDNBnorm = zz['Total_Data/ERFormmoon'][:]


fillarray=np.zeros([2816,3584,12]) #inputs not the truth
fillarray[:,:,0]=B_M08norm[:]
fillarray[:,:,1]=B_M09norm[:]
fillarray[:,:,2]=B_M10norm[:]
fillarray[:,:,3]=B_M11norm[:]
fillarray[:,:,4]=B_HNCCnorm[:]
fillarray[:,:,5]=B_ERFDNBnorm[:]





##################################################################
##################################################################
##################################################################
#STEP 6: Run Kyles test/train and do adjustments


##################################################################
##############POSTPROCESSING######################################
##################################################################
#STEP 7: Postprocess the data and compare
# PHD_proj_POSTPROCESS











