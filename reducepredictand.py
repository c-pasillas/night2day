{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['latitude',\n",
       " 'longitude',\n",
       " 'DNB',\n",
       " 'M12',\n",
       " 'M13',\n",
       " 'M14',\n",
       " 'M15',\n",
       " 'M16',\n",
       " 'channels',\n",
       " 'samples']"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "case=np.load('fullcase.npz')\n",
    "case.files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = case['DNB']\n",
    "y = case['M12']\n",
    "a = case['latitude']\n",
    "b = case['longitude']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(93, 3056, 3759)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = case['M12']\n",
    "a = case['latitude']\n",
    "b = case['longitude']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(93, 3056, 3759, 4)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reduced = np.stack((a,b,x,y), axis =-1)\n",
    "reduced.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "   np.savez(\"kangaroo\", myarray = reduced)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['myarray']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new = np.load('kangaroo.npz')\n",
    "new.files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(93, 3056, 3759, 4)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z = new['myarray']\n",
    "z.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['latitude',\n",
       " 'longitude',\n",
       " 'DNB',\n",
       " 'M12',\n",
       " 'M13',\n",
       " 'M14',\n",
       " 'M15',\n",
       " 'M16',\n",
       " 'channels',\n",
       " 'samples',\n",
       " 'M12norm',\n",
       " 'M13norm',\n",
       " 'M14norm',\n",
       " 'M15norm',\n",
       " 'M16norm',\n",
       " 'BTD1213',\n",
       " 'BTD1213norm',\n",
       " 'BTD1214',\n",
       " 'BTD1214norm',\n",
       " 'BTD1215',\n",
       " 'BTD1215norm',\n",
       " 'BTD1216',\n",
       " 'BTD1216norm',\n",
       " 'BTD1312',\n",
       " 'BTD1312norm',\n",
       " 'BTD1314',\n",
       " 'BTD1314norm',\n",
       " 'BTD1315',\n",
       " 'BTD1315norm',\n",
       " 'BTD1316',\n",
       " 'BTD1316norm',\n",
       " 'BTD1412',\n",
       " 'BTD1412norm',\n",
       " 'BTD1413',\n",
       " 'BTD1413norm',\n",
       " 'BTD1415',\n",
       " 'BTD1415norm',\n",
       " 'BTD1416',\n",
       " 'BTD1416norm',\n",
       " 'BTD1512',\n",
       " 'BTD1512norm',\n",
       " 'BTD1513',\n",
       " 'BTD1513norm',\n",
       " 'BTD1514',\n",
       " 'BTD1514norm',\n",
       " 'BTD1516',\n",
       " 'BTD1516norm',\n",
       " 'BTD1612',\n",
       " 'BTD1612norm',\n",
       " 'BTD1613',\n",
       " 'BTD1613norm',\n",
       " 'BTD1614',\n",
       " 'BTD1614norm',\n",
       " 'BTD1615',\n",
       " 'BTD1615norm',\n",
       " 'DNBfix',\n",
       " 'DNB_norm',\n",
       " 'DNB_full_moon_norm',\n",
       " 'DNB_log_norm',\n",
       " 'DNB_log_full_moon_norm',\n",
       " 'DNB_log_Miller_full_moon']"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "case=np.load('fullcase_normalized.npz')\n",
    "case.files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = case['DNB_full_moon_norm']\n",
    "y = case['M15norm']\n",
    "a = case['latitude']\n",
    "b = case['longitude']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "yoon_reduced = np.stack((a,b,x,y), axis =-1)\n",
    "yoon_reduced.shape\n",
    "np.savez(\"yoon_norm_M15\", myarray = yoon_reduced)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['myarray']"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new = np.load('yoon_norm_M15.npz')\n",
    "new.files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(93, 3056, 3759, 4)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z = new['myarray']\n",
    "z.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Xtrain = z[\"2\"][:10,:,:,0].flatten()\n",
    "Xtest = z['2'][:2,:,:,0].flatten()\n",
    "Ytrain = z['3'][:10].flatten()\n",
    "Ytest = z[\"3\"][:2].flatten()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
