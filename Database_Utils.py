''' Authour: Callum Davidson
Purpose: This is a library of useful python tool for handling data withing the Cass Weather Groups's database.
'''
import pandas as pd
import os
import datetime as dt
import warnings
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.impute as skimp
import sklearn.linear_model as sklrm
import sklearn.metrics as mtrcs
import sys
import sklearn.neighbors._base
from sklearn.experimental import enable_iterative_imputer
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
import itertools



def Combine_All_Years(database_path, datasets = None, allowed_stations = None):
    '''This is a function that combines all yearly Formatted, Filtered or Cleaned .csv files for each station into one single .csv.
    
    :param str database_path: the path to the database
    :param list datasets: a list of datasets to combine ('Formatted', 'Filtered' or 'Cleaned'). Leave as None to collect all stations.
    :param list datasets: a list of stations to combine. Leave as None to collect all stations.
    '''
    os.chdir(database_path)
    if not bool(datasets):
        datasets = [f for f in os.listdir() if f.endswith('_Data') and not f.startswith('Raw')]

    for dataset in datasets:
        try:
            os.chdir(dataset)
        except:
            raise ValueError('Unrecognised dataset path: {dataset}')
        
        stations = allowed_stations
        if not bool(stations):
            stations = [f for f in os.listdir() if f.endswith('_Hourly') or f.endswith('_Daily')and not f.startswith('.')]
            
        for station in stations:
            try:
                os.chdir(station)
            except:
                raise ValueError('Unrecognised station path: {dataset}')
                
            print('Combining ' + dataset+': '+station+'             ', end='\r')
            files = [f for f in os.listdir() if f.endswith('.csv')]
            out = pd.DataFrame()

            for f in files:
                out = pd.concat([out,pd.read_csv(f)])
                
            out = out[~out.Time.duplicated(keep='first')] 
            os.chdir('..')
            out.to_csv(station+'.csv', index=False)

        os.chdir('..')

def update_metadata(database_path):
    '''Updates an existing metadata .csv file to include most current data availability for each station. Returns the updated metadata DataFrame.
    
    :param str database_path: the database directory, containing a Metadata.csv file with sation names and information.
    '''
    os.chdir(database_path)
    metadata = pd.read_csv('Metadata.csv')
    
    os.chdir('Formatted_Data')
    
    stations = metadata.Station_Name
    metadata.index = metadata.Station_Name
    
    for station in stations:
        try:
            data = pd.read_csv(station+'.csv')
        except:
            print(f'{station} not in database.')
            continue
        if station.endswith('Hourly'):
            data.Time = pd.to_datetime(data.Time, format='%Y-%m-%d %H:%M:%S')
        elif station.endswith('Daily'):
            try:
                data.Time = pd.to_datetime(data.Time, format='%Y-%m-%d %H:%M' %p)
            except:
                try:
                    data.Time = pd.to_datetime(data.Time, format='%Y-%m-%d %H:%M')
                except:
                    try:
                        data.Time = pd.to_datetime(data.Time, format='%Y-%m-%d')
                    except:
                        raise ValueError(f"{station}'s datetime format is not recognised")
        print(f'Updated station: {station}')
        tmin = data.Time.min()
        tmax = data.Time.max()
        metadata.loc[station, 'Time_Available'] = f'{tmin}, {tmax}'
        metadata.loc[station, 'Data_Available'] = str(data.columns.to_list())
    metadata.to_csv(database_path+'/Metadata.csv',index=False)
    os.chdir(database_path)
    return metadata

def collect_variables(stations, variables, ymin, ymax, timestep, database_path, output_path=None):
    '''This collects a range of specified variables for a range of weather station datasets and merges them into one dataframe. Each
    variable is labelled as <VarName>_<StationName> in the output dataset. Saves the dataset as a .csv into the path specified.
    
    :param list stations: a list of ordered tuples containing (StationName, Dataset) where dataset is one of 
    {'Filtered','Cleaned','Formatted'}.
    :param list variables: a list of data varaibles to be collected from each station. 
    :param int ymin: year to begin dataset at
    :param int ymax: year to end dataset
    :param str timestep: timestep of the data, on of {'Hourly','Daily'}, does not support multiple timesteps
    :param str database_path: path to the weather station database
    :param str output_path: place to store output dataframe.csv
    '''
    years = range(ymin, ymax+1)
    total_range= pd.DataFrame()
    if type(stations) is str:
        stations = [stations]
    if type(variables) is str:
        variables = [variables]
    
    for year in years:
        out = pd.DataFrame()
        for station in stations:
            print(f'{station}: {year}                 ', end='\r')
            try:
                data = pd.read_csv(database_path+'/'+station[1]+'_Data/'+station[0]+'_'+timestep+'/'+station[0]+'_'+str(year)+'_'+timestep+'.csv')
            except:
                continue
            out['Time']=data['Time']
            for var in variables:
                try:
                    values = data[var]
                    out[var+'_'+station[0]] = values
                except:
                    pass
        total_range = pd.concat([total_range,out], ignore_index=True)
    print('\n')
    total_range=total_range[~total_range.Time.duplicated(keep='first')]
    total_range = total_range.dropna(how='all',axis=1)
    total_range.index = pd.to_datetime(total_range.Time)
    total_range.drop('Time',axis=1,inplace=True)
    if output_path is not None:
        total_range.to_csv(output_path+'/'+str(years[0])+'_'+str(years[-1])+'.csv', index=False)
    return total_range

def mask_outside(data, bounds):
    ''' This takes a pandas series and masks any values that fall outside of the bounds given.
    :param data series: a pandas series to mask
    :param tuple bounds: (lowerBound, upperBound)
    '''
    lower, upper = bounds
    data = data.mask(data< lower)
    data = data.mask(data> upper)
    return data

def Parse_Cliflo(datapath, Parsed_Dir, Filtered_Dir, Columns, bounds):
    '''Parses raw Cliflo data (comma delimited) into formatted and filtered .csv files.
    :param str datapath: path of cliflo raw file
    :param str Parsed_Dir: path of formatted data
    :param str Filtered_Dir: path of filtered data
    :param pandas.DataFrame Columns: Column dictionary (from file)
    :param dict bounds: a dictionary containing the desired bounds for each data variable {VarName: (lowerBound, upperBound), ...}
    '''
    data  = pd.read_csv(datapath)

    time_var = 'Day(Local_Date)'
    try:
        data[time_var] = pd.to_datetime(data[time_var],format='%Y%m%d:%H%M')
    except:
        try:
            time_var = 'Date(NZST)'
            data[time_var] = pd.to_datetime(data[time_var],format='%Y%m%d:%H%M')
        except:
            raise KeyError('Cannot identify time variable in '+datapath)
            
    years = range(min(pd.to_datetime(data[time_var],format='%Y%m%d:%H%M')).year,
                  max(pd.to_datetime(data[time_var],format='%Y%m%d:%H%M')).year+1)
    obs_hour = min(pd.to_datetime(data[time_var],format='%Y%m%d:%H%M')).hour
    for y in years:
        print(y, end='\r')

        out_data = pd.DataFrame()
        out_data.index = pd.date_range(dt.datetime(y,1,1),dt.datetime(y,12,31))
        data.index = data[time_var]
        data.index.names = ['Time']

        out_data = pd.DataFrame()
        out_data.index = pd.date_range(dt.datetime(year=y,month=1,day=1,hour=obs_hour),
                                       dt.datetime(year=y+1,month=1,day=1,hour=obs_hour),freq='D')

        # Create container for filtered data
        filtered_data = pd.DataFrame()
        filtered_data.index = pd.date_range(dt.datetime(year=y,month=1,day=1,hour=obs_hour),
                                            dt.datetime(year=y+1,month=1,day=1,hour=obs_hour),freq='D')

        # Attach data to out_data
        for C in Columns.columns[4:]:
            for c in Columns[C]:
                if c in list(data.columns):
                    try:
                        # See if value can be interpreted as numeric
                        out_data[C] = pd.to_numeric(data[c], errors='coerce')
                        break
                    except:
                        pass

        # Filter and attach data to filtered_data
        for C in Columns.columns[4:]:
            for c in Columns[C]:
                if c in list(data.columns):
                    try:
                        # See if value can be interpreted as numeric
                        filtered_data[C] = mask_outside(pd.to_numeric(data[c], errors='coerce'), bounds[C])
                        break
                    except:
                        pass

        out_data.index.names = ['Time'] # Set Index name to Time
        out_data.to_csv(Parsed_Dir+'_'+str(y)+'_Daily.csv') # Save to file

        filtered_data.index.names = ['Time'] # Set Index name to Time
        filtered_data.to_csv(Filtered_Dir+'_'+str(y)+'_Daily.csv') # Save to file
        
    return None

def cleanHourly_dirtyDaily(hourly_path, daily_path, station_name, out_path, years, Columns, bounds):
    '''This takes a cleaned hourly dataset and syncs it with a uncleaned daily dataset. Any variable with missing data in the 
    corresponding daily in the hourly dataset is simply deleted.
    :param str hourly_path: path of hourly dataset
    :param str daily_path: path of daily dataset
    :param str station_name: name of station dataset
    :param range years: years to clean
    :param pandas.DataFrame Columns: Column dictionary (from file)
    :param dict bounds: a dictionary containing the desired bounds for each data variable {VarName: (lowerBound, upperBound), ...}
    '''

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    for y in years:
        print(y,end='\r')
        cleaned_data = pd.read_csv(hourly_path+station_name+'_'+str(y)+'_Hourly.csv')
        data = pd.read_csv(daily_path+station_name+'_'+str(y)+'_Daily.csv')

        data.Time = pd.to_datetime(data.Time,format='%Y-%m-%d')
        cleaned_data.Time = pd.to_datetime(cleaned_data.Time)
        out_data = data.copy()

        for i, day in enumerate(data.Time):
            hours = pd.date_range(day+dt.timedelta(hours=-23), day+dt.timedelta(hours=0),freq='H').tolist()
            day_data = cleaned_data[cleaned_data.Time.isin(hours)]
            
            for var in cleaned_data.columns.tolist():
                daily_vars = [c for c in data.columns if c.startswith(var)]
                
                if any(day_data[var].isna()): # if any of the hourly data in the last day is missing
                    for daily_var in daily_vars:
                        out_data.loc[i,daily_var]=np.nan

        out_data.to_csv(out_path+'/'+station_name+'_'+str(y)+'_Daily.csv',index=False)
    
    return None
        
def find_extent(points, buffer=0.1):
    '''
    Creates a extent around a given set of points with a buffer.
    :param np.ndarray points: list of np.arrays with columns containing longitude and latitude coords.
    :param float buffer: ratio of blank space to allow on border of map, as a ratio of the length of the polygon in lat/lon
    '''
    
    lon_max=np.max(points[:,1])
    lon_min = np.min(points[:,1])
    lat_max = np.max(points[:,0])
    lat_min=np.min(points[:,0])

    lon_buffer = abs(lon_max-lon_min)*buffer
    lat_buffer = abs(lon_max-lon_min)*buffer
    
    top = lon_max+lon_buffer
    bottom = lon_min-lon_buffer
    left = lat_min-lat_buffer
    right = lat_max+lat_buffer
    
    return [ bottom, top, left, right]


def map_stations_and_corrs(data, metadata, target_station, map_zoom, figsize=(10,10), figpath=None, marker_scale_factor=100, annotation_text_size=10, dpi=200, timestep='Daily', extent_pad=0.1, legend_labelspacing = 1, legend_markerscale=0.5, legend_loc = 'center left'):
    ''' This plots the location and correlation (relative to a target_station) stations on a map. 
    :param pandas.DataFrame data: the model data
    :param pandas.DataFrame metadata: metadata from metadata.csv in database
    :param str target_station: The station to which other station's correlation scores are measured.
    :param int mapzoom: The zoom of the basemap, if in doubt; start small and get larger until desired result is reached. 11 is a good level for
    the cass stations.
    :param float marker_scale_factor: the scale factor for the marker size
    :param float annotation_text_size: the scale factor for the annotation text size
    :param str figpath: The path to save the output figure to, leave as None and the figure will not be saved.
    :param float dpi=200: The dpi of the saved figure
    :param str timestep: The timestep of the data, normally leaving timestep='Daily' is fine unless station is only available at one timestep.
    :param float extent_pad: the map padding around the outermost points 
    :param float legend_labelspacing: the spacing between labels on the legend
    :param float legend_markerscale: the scale factor of markers as display on the legend
    :param str legend_loc: the location of the legend on the axes, see matplotlib docs for allowed values
    '''
    
    basemap = cimgt.Stamen('terrain-background')
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=basemap.crs))
    stations = list(set([c.split('_')[-1] for c in data.columns]))
    extent = find_extent(np.array([[metadata.loc[f'{station}_{timestep}','Latitude'], metadata.loc[f'{station}_{timestep}','Longitude']] for station in stations]), extent_pad) # finds (xmin, xmax, ymin, ymax)
    ax.set_extent(extent, crs=ccrs.Geodetic())
    correlations = data.corr()[[c for c in data.columns if c.endswith(f'_{target_station}')]]
    
    ax.add_image(basemap, map_zoom)
    c_arr = []
    annotations = []
    
    
    for station in stations:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            correlation = np.mean(correlations.loc[[c for c in correlations.index if c.endswith(f'_{station}')]].values)
        if np.isnan(correlation):
            correlation = 0
        
        if station != target_station:
            size = np.exp(abs(correlation)*4)*marker_scale_factor
            ax.scatter(metadata.loc[f'{station}_{timestep}','Longitude'],
                       metadata.loc[f'{station}_{timestep}','Latitude'],
                       transform=ccrs.PlateCarree(),
                       label = station+' mean correlation'+': % .2f'%correlation,
                       edgecolor='k',s=size)
            transform = transform = ccrs.PlateCarree()._as_mpl_transform(ax)
            annotations.append(ax.annotate('% .2f'% correlation, (metadata.loc[f'{station}_{timestep}','Longitude'], 
                                metadata.loc[f'{station}_{timestep}','Latitude']),
                                xycoords=transform, size=annotation_text_size,ha='center', va='center'))
        else:
            size = marker_scale_factor
            marker = 'x'
            ax.scatter(metadata.loc[f'{station}_{timestep}','Longitude'],
                       metadata.loc[f'{station}_{timestep}','Latitude'],
                       transform=ccrs.PlateCarree(),
                       label = station+': Target Station' , color='k',
                       edgecolor='k',s=size)
        
    ax.legend(labelspacing=legend_labelspacing, loc=legend_loc, bbox_to_anchor=(1, 0.5))
    if bool(figpath):
        plt.savefig(figpath+'/'+target_station+'_corrMap.png',dpi=dpi)
    return plt.show()



def check_missingness(data, missing_threshold=.5):
    ''' Many imputation methods suffer from instability when the missingness rate of the dataset or individual variables is too high.
    This function checks each variable and warns the user about which variables may be the cause of any unexpected or poor results seen
    in the final imputation. Returns the ratio of missing data to data in the dataframe.
    :params pandas.DataFrame data: The umimputed data
    :params float missing_threshold: The missingness threshold, 50% by defualt. 
    '''
    for var in data.columns:
        if air_data.isna().sum()[var]/len(air_data[var]) > missing_threshold:
            print(f'Warning! Variable {var} is missing {air_data.isna().sum()[var]/len(air_data[var])*100}% of data! This might lead to imputation instability.')
    if air_data.isna().sum().sum()/(len(air_data)*len(air_data.columns))> missing_threshold:
        print(f'Warning! Dataset is missing {air_data.isna().sum()[var]/len(air_data[var])*100}% of data! This is likely to lead to imputation instability.')
        
    return air_data.isna().sum().sum()/(len(air_data)*len(air_data.columns))

def split_test_train(data, samples, stations=None, variables=None,random_state=None):
    ''' This function pops a random sample of data out of the dataset for use in imputation validation.
    :param DataFrame data:
    :param int samples: The number of samples for each variable being assessed
    :param list stations: a list of station names to sample, leave as None if all are desired.
    :param list variables: a list of variables to sample, leave as None if all are desired
    '''
    test_data = pd.DataFrame(index=data.index)
    train_data = data.copy()
    
    if not bool(stations):
        stations = list(set([c.split('_')[-1] for c in data.columns]))
    if not bool(variables):
        variables = list(set(['_'.join(c.split('_')[:-1]) for c in data.columns]))
    
    for station in stations:
        for var in variables:
            try:
                sample = data[var+'_'+station][~data[var+'_'+station].isna()].sample(n=samples,random_state=random_state)
                test_data[var+'_'+station] = sample
                train_data.loc[sample.index, var+'_'+station] =np.nan
            except:
                pass
            
    return test_data, train_data

def create_blackout(data, stations=None, variables=None, holes=[[pd.Timestamp(2010,7,1,9,0,0),pd.Timestamp(2010,11,1,9,0,0)]]):
    '''This is a special case of a test/train data splitter. Rather than selecting data at random, data from a station is deleted across a 
    time range, this simulating a station blackout. The statistical difference is quite nuanced, however this is an important evaluation since this
    is the way most of our data is lost in weather stations. To summarise, this creates data Missing At Random (MAR), rather than Missing 
    Completely At Random (MCAR). Furthermore, this function can remove all variables from that station (complete backout) thus eliminating the 
    possibility of using closely correlated station variables to fill holes. 
    :params pandas.DataFrame data: Dataset to poke holes in
    :params iterable stations: A list of stations to remove data from, leave as None to remove data from all stations.
    :params iterable variables: A list of variables to delete from the stations specified,  leave as None to remove data from variables.
    :params iterable holes: A list of pd.Timestamp tuples [(t1,t2),(t3,t4)...] specifying the start and end of the holes you wish to create.
    '''
    train_data = data.copy()
    test_data = pd.DataFrame(index=data.index,columns=train_data.columns)
    if type(stations) is str:
            stations = [stations]
    if type(variables) is str:
            variables = [variables]
    if type(holes[0]) is not list:
        holes = [holes]
    if stations is None:
        stations = list(set([station.split('_')[-1] for station in data.columns]))
    if variables is None:
        variables = list(set(['_'.join(variable.split('_')[:-1]) for variable in data.columns]))
    
    for i, hole in enumerate(holes): 
        hole[0] = data.index[data.index.get_loc(hole[0], method='nearest')]
        hole[1] = data.index[data.index.get_loc(hole[1], method='nearest')]
        holes[i] = hole
    print(holes)
    
    for hole in holes:
        for station in stations:
            for var in [v for v in train_data.columns if v.endswith(station)]:
                print([v in var for v in variables])
                if any([v in var for v in variables]):
                    print(var)
                    test_data.loc[hole[0]:hole[1],var] = train_data.loc[hole[0]:hole[1],var]
                    train_data.loc[hole[0]:hole[1],var]=np.nan
                elif variables is None:
                    print('asda')
                    test_data.loc[hole[0]:hole[1],var] = train_data.loc[hole[0]:hole[1],var]
                    train_data.loc[hole[0]:hole[1],var]=np.nan
                    
    return train_data,test_data

def plot_continuity(data,figsize=(15,8),figpath = None):
    cont = data.copy()
    for i,c in enumerate(data.columns):
        cont[c][~cont[c].isna()]=i
    fig=plt.figure(figsize=figsize)
    plt.plot(cont,label=data.columns,linewidth=4, c='b')
    plt.yticks(ticks = range(i+1), labels=data.columns)
    plt.grid()
    plt.tight_layout()
    if bool(figpath):
        plt.savefig(figpath,dpi=200)
    plt.show()
    return None



def combine_and_drop(data, combine, drop, how='keep'):
    '''Some Cliflo datasets contain the same data but for whatever reason have different avaiabilities depending 
    on the data specified. This function combines two columns of a dataframe and drops one of the unneeded columns
    
    :param pandas.DataFrame data:  The DataFrame
    :param str combine: the column along which the data will be merged
    :param str drop: the column to drop from the DataFrame
    :param str how: How the data is combined. If "keep" then the data from "drop" is only used to fill in missing values in "combine". 
    If "replace" then data from "drop" replaces data from "combine". Default is "keep."'''
    
    if how not in {'keep','replace'}:
        raise ValueError('param how must be one of "keep", "replace".')
    data = data.copy()
    if how=='keep':
        for var in ['_'.join(v.split('_')[:-1])+'_' for v in data.columns if v.endswith(combine)]:
            data[var+drop][~data[var+combine].isna()]=data[var+combine][~data[var+combine].isna()]
            data[var+combine] = data[var+drop]
            data.drop(var+drop,axis=1,inplace=True)
    else:
        for var in ['_'.join(v.split('_')[:-1])+'_' for v in data.columns if v.endswith(combine)]:
            data[var+combine][~data[var+drop].isna()] = data[var+drop][~data[var+drop].isna()]
            data.drop(var+drop,axis=1,inplace=True)
    return data


def transformTemps_900_to_000(data, station):
    '''Some temperature data is only available at 0900 hrs, it is fairly trivial to convert this to 0000 format
    by simply moving the max temperature back by 24hrs, this function does exactly that. Note that this transformation is not perfect.
    But it does get you very close 99% of the time'''
    
    data = data.copy()
    
    data['Air_Temp_Max_'+station] = data['Air_Temp_Max_'+station].shift(-1)
    
    return data

def transformTemps_000_to_900(data, station):
    '''Some temperature data is only available at 0000 hrs, it is fairly trivial to convert this to 0900 format
    by simply moving the max temperature forward by 24hrs, this function does exactly that. Note that this transformation is not perfect.
    But it does get you very close 99% of the time.'''
    
    data = data.copy()
    
    data['Air_Temp_Max_'+station] = data['Air_Temp_Max_'+station].shift(1)
    
    return data

def plot_imputation_evaluation(test_data,imputed_data, stations=None, variables=None, figpath = None):
    '''Evaluates a test dataset against estimations produced by an imputation. Displays results on a plot.
    :param pandas.DataFrame test_data: A dataset containing test values.
    :param pandas.DataFrame imputed_data: The imputed dataset
    :param list variables: a list containing vairables to analyse. Leave as None to examine all variables
    :param str figpath: a directory to store output in, leave as none to not save the figure 
    
    '''
    
    
    
    if not bool(variables):
        variables=imputed_data.columns
    if bool(stations):
        variables = [v for v in variables if v.endswith(tuple(stations))]
    
    score_arr = np.zeros([4,len(variables)])
    
    for i, c in enumerate(variables):
        pred = imputed_data.loc[test_data[c][~test_data[c].isna()].index,c]
        observed = test_data[c][~test_data[c].isna()]


        print(f'{c} R squared: {mtrcs.r2_score(pred,observed)}')
        print(f'{c} MSE: {mtrcs.mean_squared_error(pred,observed)}')
        print(f'{c} MAE: {mtrcs.mean_absolute_error(pred,observed)}')
        print(f'{c} Mean Error/Bias: {sum(pred-observed)/len(observed)}')
        score_arr[0,i]=mtrcs.r2_score(pred,observed)
        score_arr[1,i]=mtrcs.mean_squared_error(pred,observed)
        score_arr[2,i]=mtrcs.mean_absolute_error(pred,observed)
        score_arr[3,i]=sum(pred-observed)/len(observed)
    score_arr_norm = score_arr.copy()
    score_arr_norm[0,:] = 1-score_arr_norm[0,:]/max(abs(score_arr_norm[0,:]))
    score_arr_norm[1,:] = abs(score_arr_norm[1,:]/max(abs(score_arr_norm[1,:])))
    score_arr_norm[2,:] = abs(score_arr_norm[2,:]/max(abs(score_arr_norm[2,:])))
    score_arr_norm[3,:] = abs(score_arr_norm[3,:]/max(abs(score_arr_norm[3,:])))
    
    # Plot figure...
    fig =plt.figure(figsize=(20,5))
    im = plt.imshow(score_arr_norm, cmap='RdPu')
    plt.yticks(range(4),labels=['R$^2$','MSE','MAE','Mean Bias'])
    plt.xticks(range(len(imputed_data.columns)),labels=imputed_data.columns,rotation=90)
    
    for x in range(len(imputed_data.columns)):
        for y in range(4):
            plt.annotate(np.round(score_arr[y,x],2),(x,y),ha='center',va='center')

    cbar = plt.colorbar(im,fraction=0.046)
    cbar.set_ticks([0,1])
    cbar.set_ticklabels(['Best', 'Worst'])
    
    if bool(figpath):
        plt.savefig(figpath, dpi=200)
        
    plt.show()
    return None

def pre_impute(data, stations, model=None, imputer=None, target=None, how='keep_only'):
    ''' High missingness in datasets can introduce instability into the final imputation. In some instances, it can be benificial to
    impute small gaps in data between very similar stations. Typically, this is only recommended for stations which are closely co-located.
    This function imputes results for a subset of columns in a dataset. 
    :param pandas.DataFrame data: The dataframe containing the columns used in the preimpute
    :param list stations: The stations from the dataframe used in the pre_impute
    :param object model: a model from the sci-kit learn package, defaults to LinearRegression
    :param object imputer: imputer class to use for missing data
    :param str target: The target station, if set all other stations will be dropped from the dataframe. Leave as none to keep all data.
    :param str how: What to do with the target and non-target data? If "keep_only" then non-target data is dropped from the data frame. If 
    "keep_raw" non-target data is kept in its raw format. If all imputations are to be kept, set target station to None.
    :returns: A dataset with values imputed in columns specified.
    '''
    
    if how not in {'keep_only','leave_raw'}:
        raise ValueError('parameter how must be one of "keep_only", "leave_raw"')
    data = data.copy()
    columns = [c for c in data.columns if c.endswith(tuple(stations))]
    imp_data = data[columns].copy()
    imp_data.dropna(axis=0,how='all',inplace=True)
    
    if not bool(model): # If there is no model specified use linear regression
        model = sklrm.LinearRegression()
    if not bool(imputer): # If there is no imputer use Scikit-learn iterative imputer
        imputer = skimp.IterativeImputer(estimator=model,
                                       missing_values=np.nan, 
                                       max_iter=3000, 
                                       verbose=2,
                                       random_state=2, 
                                       initial_strategy = 'mean',
                                       imputation_order='arabic',
                                       max_value=50,
                                       min_value=-20)
    imp_data = pd.DataFrame(imputer.fit_transform(imp_data[columns]), index=imp_data.index, columns=columns)
    
    if bool(target): # If there is a target station drop all other stations from dataset
        if target not in [c.split('_')[-1] for c in columns]:
            raise ValueError('Target station not in columns.')
            
        for c in [c for c in columns if c.endswith(target)]:
            columns.remove(c)
            
        imp_data.drop(columns, axis=1, inplace=True)
        if how=='keep_only':
            data.drop(columns, axis=1, inplace=True)
            
    for c in imp_data.columns:
        data[c] = imp_data[c]
        
    return data


def map_stations(data, metadata, map_zoom, figsize = (10,10), figpath=None, markersize=100, annotation_text_size=10, dpi=200, timestep='Daily',\
                 extent_pad=0.1, legend_labelspacing = 1, legend_markerscale=0.5, legend_loc = 'center left'):
    ''' This plots the location of stations on a map. 
    :param pandas.DataFrame data: the model data
    :param pandas.DataFrame metadata: metadata from metadata.csv in database
    :param str target_station: The station to which other station's correlation scores are measured.
    :param int mapzoom: The zoom of the basemap, if in doubt; start small and get larger until desired result is reached. 11 is a good level for
    the cass stations.
    :param float marker_scale_factor: the scale factor for the marker size
    :param float annotation_text_size: the scale factor for the annotation text size
    :param str figpath: The path and name to save the output figure to, leave as None and the figure will not be saved.
    :param float dpi=200: The dpi of the saved figure
    :param str timestep: The timestep of the data, normally leaving timestep='Daily' is fine unless station is only available at one timestep.
    :param float extent_pad: the map padding around the outermost points 
    :param float legend_labelspacing: the spacing between labels on the legend
    :param float legend_markerscale: the scale factor of markers as display on the legend
    :param str legend_loc: the location of the legend on the axes, see matplotlib docs for allowed values
    '''
    
    basemap = cimgt.Stamen('terrain-background')
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=basemap.crs))
    stations = list(set([c.split('_')[-1] for c in data.columns]))
    extent = find_extent(np.array(
        [[metadata.loc[f'{station}_{timestep}','Latitude'], 
          metadata.loc[f'{station}_{timestep}','Longitude']] 
         for station in stations]),
                         extent_pad) # finds (xmin, xmax, ymin, ymax)
    marker = itertools.cycle(('o', 'D', 's', '^', 'v', '*','X', 'P', 'd'))
    ax.set_extent(extent, crs=ccrs.Geodetic())    
    ax.add_image(basemap, map_zoom)
    for station in stations:
        ax.scatter(metadata.loc[f'{station}_{timestep}','Longitude'],
                   metadata.loc[f'{station}_{timestep}','Latitude'],
                   transform=ccrs.PlateCarree(),
                   label = station,
                   edgecolor='k',s=markersize,marker=next(marker)
                   )

    ax.legend(labelspacing=legend_labelspacing, loc=legend_loc, bbox_to_anchor=(1, 0.5))
    if bool(figpath):
        plt.savefig(figpath,dpi=dpi)
    return plt.show()


class impute_multiple():
    '''This is a class that handles imputing multiple solutions to a dataset using various samples and random seeds. This is advantagous because
    it allows you to increase the number of samples without affecting impute quality. For instance, 20 inputes with a sample size of 10 is a 200
    datum sample.
    '''
    def __init__(self, data, imputer, n_datasets=4, random_seeds=None):
        self.n_datasets = n_datasets
        self.random_seeds = random_seeds
        self.imputer = imputer
        self.datasets=[]
        self.data = data
        
        if not bool(random_seeds):
            self.random_seeds = np.random.randint(999999, size=n_datasets)
                                                  
        if random_seeds is not None and len(random_seeds)!=n_datasets:
            raise ValueError('Length of random_seeds does not match the number of datasets!')
        return None
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()
        
    def split_tests_trains(self, n_samples=None, stations=None, variables=None, random_states=None):
        ''' Creates MCAR testing and training data for each dataset using split_test_train function.
        :param int n_samples: Number of samples to take from each varible, default is None and if None then number of samples will be
        calculated by length_of_dataset * 0.005 (i.e. 0.5% of total data used to test imputation.)
        :param list stations: Stations to sample
        :param list variables: Variables to sample
        :param int, list, tuple random_states: if random_states=None, a completely random sample is taken each time. If int, then the same
        sample is to be taken from every station. If tuple or list then random states are taken (in order) from the list.
        '''
        self.testing_datasets = []
        self.training_datasets = []
        
        if n_samples is None:
            n_samples = int(self.data.shape[0]*0.005)
        
        for i in range(self.n_datasets):
            if random_states is None:
                random_states = self.random_seeds
            if type(random_states) is int:
                random_state = random_states
            elif type(random_states) is list or tuple:
                if len(random_states) != self.n_datasets:
                    raise ValueError('Number of random states must match number of datasets!')
                random_state = random_states[i]
            
            test, train = split_test_train(self.data, n_samples, stations=stations, variables=stations, random_state=random_state)
            self.testing_datasets.append(test)
            self.training_datasets.append(train)
            
    def fit_transform(self, training_data=False, **kwargs):
        '''Trains and fits imputation model to datasets with random states specified by random_seeds. If no training_datasets are 
        provided, imputation will simply be fit to the raw data. 
        '''
        for i in range(self.n_datasets):
            print(f"Imputation {i+1} of {self.n_datasets} with seed {self.random_seeds[i]}")
            self.imputer.random_state = self.random_seeds[i]
            if training_data is False:
                self.datasets.append(pd.DataFrame(self.imputer.fit_transform(self.data, **kwargs), 
                                                  index=self.data.index, 
                                                  columns=self.data.columns))
            elif type(self.training_datasets) is list:
                self.datasets.append(pd.DataFrame(self.imputer.fit_transform(self.training_datasets[i], **kwargs), 
                                                  index=self.training_datasets[i].index, 
                                                  columns=self.training_datasets[i].columns))
            else:
                 self.datasets.append(pd.DataFrame(self.imputer.fit_transform(self.training_datasets, **kwargs), 
                                                  index=self.training_datasets.index, 
                                                  columns=self.training_datasets.columns))
                    
    def collect_imputations(self, only_missing=False):
        '''Collects all imputed datasets into one dataframe with columns labelled Var_Name_1, Var_Name_2... Var_Name_n.4
        :params bool only_missing: If True, only imputed data will be collected. Default is False.
        '''
        self.all_imps = pd.DataFrame(index = self.data.index)
        for c in self.data.columns:
            mask = self.data.index[self.data[c].isna()] # Collect index where data is missing (imputed vals only)
            for i, dataset in enumerate(self.datasets):
                if only_missing:
                    self.all_imps[c+'_'+str(i+1)] = dataset.loc[mask,c]
                else:
                    self.all_imps[c+'_'+str(i+1)] = dataset[c] # Collect all vals if not only_missing
        return None
    

                
    def plot_evaluation(self, imputed_datasets = None, stations=None, variables=None, figpath = None):
        '''Evaluates a test dataset against estimations produced by an imputation. Displays results on a plot.
        :param pandas.DataFrame test_data: A dataset containing test values.
        :param pandas.DataFrame imputed_data: The imputed dataset
        :param list variables: a list containing vairables to analyse. Leave as None to examine all variables
        :param str figpath: a directory to store output in, leave as none to not save the figure 
        '''
        if stations is None:
            if type(self.testing_datasets) is list:
                columns = (self.testing_datasets[0].dropna(axis=1, how='all')).columns
            else:
                columns = (self.testing_datasets.dropna(axis=1, how='all')).columns
            stations = list(set([station.split('_')[-1] for station in columns]))
        if variables is None:
            variables=self.data.columns
        if stations is not None:
            variables = [v for v in variables if v.endswith(tuple(stations))]
        scores = []
        
        
        score_arr = np.zeros([4,len(variables)])
        if imputed_datasets is None:
            imputed_datasets = self.datasets
        
        for i, c in enumerate(variables):
            pred = []
            observed = []
            for j, dataset in enumerate(imputed_datasets):
                if type(self.training_datasets) is list:
                    pred+=dataset.loc[self.testing_datasets[j][c][~self.testing_datasets[j][c].isna()].index,c].to_list()
                    observed+=self.testing_datasets[j][c][~self.testing_datasets[j][c].isna()].to_list()
                else: 
                    pred=dataset.loc[self.testing_datasets[c][~self.testing_datasets[c].isna()].index,c].to_list()
                    observed=self.testing_datasets[c][~self.testing_datasets[c].isna()].to_list()
            pred = np.array(pred)
            observed = np.array(observed)
            
            print(f'{c} R squared: {mtrcs.r2_score(pred,observed)}')
            print(f'{c} MSE: {mtrcs.mean_squared_error(pred,observed)}')
            print(f'{c} MAE: {mtrcs.mean_absolute_error(pred,observed)}')
            print(f'{c} Mean Error/Bias: {sum(pred-observed)/len(observed)}')
            score_arr[0,i]=mtrcs.r2_score(pred,observed)
            score_arr[1,i]=mtrcs.mean_squared_error(pred,observed)
            score_arr[2,i]=mtrcs.mean_absolute_error(pred,observed)
            score_arr[3,i]=sum(pred-observed)/len(observed)
        
        score_arr_norm = score_arr.copy()
        score_arr_norm[0,:] = 1-score_arr_norm[0,:]/max(abs(score_arr_norm[0,:]))
        score_arr_norm[1,:] = abs(score_arr_norm[1,:]/max(abs(score_arr_norm[1,:])))
        score_arr_norm[2,:] = abs(score_arr_norm[2,:]/max(abs(score_arr_norm[2,:])))
        score_arr_norm[3,:] = abs(score_arr_norm[3,:]/max(abs(score_arr_norm[3,:])))

        # Plot figure...
        fig =plt.figure(figsize=(20,5))
        im = plt.imshow(score_arr_norm, cmap='RdPu')
        plt.yticks(range(4),labels=['R$^2$','MSE','MAE','Mean Bias'])
        plt.xticks(range(len(variables)),labels=variables,rotation=90)

        for x in range(len(variables)):
            for y in range(4):
                plt.annotate(np.round(score_arr[y,x],2),(x,y),ha='center',va='center')

        cbar = plt.colorbar(im,fraction=0.046)
        cbar.set_ticks([0,1])
        cbar.set_ticklabels(['Best', 'Worst'])

        if bool(figpath):
            plt.savefig(figpath, dpi=200)
            
        return plt.show()
    
    def blackout(self, holes, stations=None, variables=None):
        '''
        '''
        
        
        train, test = create_blackout(self.data, stations=stations, variables=variables, holes=holes)
        
        self.testing_datasets = [test]*self.n_datasets
        self.training_datasets = [train]*self.n_datasets
        return None
    
    
    def combine_datasets(self, datasets = None, strategy = np.mean):
        '''Combines results of all imputations.
        :param callable strategy: The method used to combine the datasets, default is numpy.mean.
        '''
        
        if datasets is None:
            datasets = self.datasets
        self.combined = pd.DataFrame(index=datasets[0].index)

        for c in datasets[0].columns:
            mean = pd.DataFrame(index=datasets[0].index)
            for i, data in enumerate(datasets):
                mean[c+str(i)] = data[c]
            self.combined[c] = mean.apply(strategy,axis=1)
        return None
    
    def dummy_indicator(self, training_data = False, variable = 'Rain', method=lambda x: np.nan if np.isnan(x) else 1*(x!=0)):
        '''Binary tree imputers sometimes struggle to capture values that contain lots of zeros, like rainfall data. One solution to this is to use 
        a binary dummy variable to determine which days are rain days. This file imputes a 
        :param bool training_data: Whether or not to create the values from the training_datasets or the raw data, defaults to False (raw data)
        :param str variable: Data variable to use for the dummy variable, defaults to Rain as that is what I have used it for. 
        :param lambda method: Creation method for dummy values. 
        :imput
        '''
        self.dummy_indicators = []
        columns = [c for c in self.data.columns if c.startswith(variable)]
        if training_data:
            indicator = pd.DataFrame(index=self.data.index)
            for dataset in self.training_datasets:
                for c in columns:
                    indicator['Is_'+c] = dataset[c].apply(method)
                self.dummy_indicators.append(indicator)
        else:
            indicator = pd.DataFrame(index=self.data.index)
            for c in columns:
                indicator['Is_'+c] = self.data[c].apply(method)
            self.dummy_indicators.append(indicator)
        return None
    
    def combine_data_dummy_indicators(self, training_data = False, variable = 'Rain', dummy_var='Is_Rain'):
        stations = [c.split('_')[-1] for c in self.data.columns if c.startswith(variable)]
        self.datasets_dummied = []
        
        for i, dataset in enumerate(self.datasets):
            data = dataset.copy()
            # data = pd.concat([data, self.imputed_indicators[i]],axis=1,join='inner')
            for station in stations:
                data[variable+'_'+station]*=self.imputed_indicators[i][dummy_var+'_'+station]
            self.datasets_dummied.append(data)
    
    def create_and_impute_dummyvars(self,dummy_imputer=None,training_data=False,impute_alongside_data=False, **imp_args):
        if dummy_imputer is None:
            self.dummy_imputer = self.imputer
        else:
            self.dummy_imputer = dummy_imputer
            
        self.dummy_indicator(training_data=training_data)
        cat_vars = np.arange(len(self.data.columns))
        self.imputed_indicators =[]
        for i in range(self.n_datasets):
            print(f"Dummy indicator imputation {i+1} of {self.n_datasets} with seed {self.random_seeds[i]}")
            self.imputer.random_state = self.random_seeds[i]
            if impute_alongside_data:
                if type(self.dummy_indicators) is list:
                    self.imputed_indicators.append(pd.DataFrame(
                                                      self.dummy_imputer.fit_transform(
                                                          pd.concat(
                                                              [self.dummy_indicators[i],self.datasets[i]], 
                                                              axis=1,
                                                              join='inner',
                                                              ignore_index=False),
                                                          cat_vars=cat_vars, **imp_args), 
                                                      index=self.dummy_indicators[i].index, 
                                                      columns=list(self.dummy_indicators[i].columns)+list(self.datasets[i].columns)))
                else:
                     self.imputed_indicators.append(pd.DataFrame(
                                                      self.dummy_imputer.fit_transform(
                                                          pd.concat(
                                                              [self.dummy_indicators[i],self.datasets[i]], 
                                                              axis=1,
                                                              join='inner',
                                                              ignore_index=False), 
                                                          cat_vars=cat_vars, 
                                                          **imp_args), 
                                                      index=self.dummy_indicators.index, 
                                                      columns=list(self.dummy_indicators[i].columns)+list(self.datasets[i].columns)))
            else:
                if type(self.dummy_indicators) is list:
                    self.imputed_indicators.append(pd.DataFrame(
                                                      self.dummy_imputer.fit_transform(
                                                          self.dummy_indicators[i],
                                                          cat_vars=cat_vars, **imp_args), 
                                                      index=self.dummy_indicators[i].index, 
                                                      columns=list(self.dummy_indicators[i].columns)))
                else:
                     self.imputed_indicators.append(pd.DataFrame(
                                                      self.dummy_imputer.fit_transform(
                                                              self.dummy_indicators[i], 
                                                          cat_vars=cat_vars, 
                                                          **imp_args), 
                                                      index=self.dummy_indicators.index, 
                                                      columns=list(self.dummy_indicators[i].columns)))
                    
                    
    def save_imputation(self, out_path, dataset = 0, impute_indicator=True):
        '''Save imputed dataset to csv file. Optional impute indicator creates a seperate column for each variable with 1 to indicate
        imputed values.
        :params str out_path: the directory and name of the file to save in form "path/file_name.csv"
        :params int or pandas.DataFrame dataset: The imputation to use, int for one of self.datasets or specify another
        :params bool impute_indicator: If true, include a binary indicator for each value to show imputed data
        '''
        if type(dataset) is int:
            try:
                dataset = self.datasets[dataset]
            except:
                raise ValueError(f'{dataset} is out of range in self.datasets')
        
        if not out_path.endswith('.csv'):
            raise ValueError('Invalid path, ensure that it is in form: "path/file_name.csv"')
            
        out = dataset.copy()
        if impute_indicator:
            for c in self.data.columns:
                out['IsImp_'+c] = 1*(self.data[c].isna())
                
        out.to_csv(out_path)
    