This is a complete database for all the resources I have collected from NIWA, ECan and the University weather stations.

------------------------------------------------- DESCRIPTION OF DIRECTORIES --------------------------------------------
- Raw_Data:

	This contains the 'raw' files in their (mostly) original formatting. These have been minimally altered by myself 
and are not all that usable, i.e. different headers, formatting, timestamps, etc. 

- Formatted_Data:
	
	This contains the raw data in a consistent format. The Timesteps are regular, variables are consistent. Each 
station has its own folder. Data is stored in .csv files by year. 

- Filtered_Data:
	
	This is a directory containing data that filtered to remove 'obviously bad' data, i.e. values outside of a 
physically feasible range. 

- Cleaned_Data:
	
	This is a special directory contained data that has been curated by hand to remove obviously spurious, but hard
to contain data. 

- SUGGESTED USE:

Generally, the formatted datasets can be used for most of the data provided by NIWA or ECan. Filtered datasets, or cleaned
datasets are strongly recommended for Chilton and Cass due to the high amount of bad data.

------------------------------------ COLUMN DICTIONARIES AND FORMATTING NOTEBOOKS-----------------------------------------

Much of the formatting of the datasets have been accomplished withing the Jupyter notebooks attached. In broad strokes, 
these contain the python scripts used to create the Cleaned, Filtered, and Formatted datasets from Raw_Data. Because these
scripts are bespoke for each differently formatted datasource, I felt it would largely be a wasted effort to turn these in
to a complete library or executable. 

The Daily_Column_Dictionary and Hourly_Column_Dictionary files contain a lookup table of all the known variable names 
across the raw datasets used. The top row are the names used in the rest of the dataset. A full breakdown of variables can 
be found below.

Changing the vairable names is as simple as editing the .csv and running the Notebooks again. 



*************************************************************************************************************************
--------------------------------------------- DAILY VARIABLE NAMES ------------------------------------------------------
*************************************************************************************************************************

Output_Code		
Year			
Day ---------------------------- These are largely unused, just a product of the datalogger at Cass and Chilton

Time --------------------------- Time in 'YYYY-mm-dd' formatting

Li190				
Li190_Max		
Li190_Max_Time		
Li200			
Li200_Max			
Li200_Max_Time ----------------- Li190 and Li200 are solar sensors for photolysis-active and total radiance respectively.

Air_Temp			
Air_Temp_Min			
Air_Temp_Min_Time		
Air_Temp_Max			
Air_Temp_Max_Time -------------- Mean, min and max air temperature readings

5m_Air_Temp			
12m_Air_Temp ------------------- Air temperature at different heights (Chilton)

Soil_Temp			
Soil_Temp_Min			
Soil_Temp_Min_Time		
Soil_Temp_Max			
Soil_Temp_Max_Time ------------- Soil Temperatures (unspecified depth)

10cm_Soil_Temp
20cm_Soil_Temp
50cm_Soil_Temp
10cm_Soil_Temp_Min
10cm_Soil_Temp_Min_Time
20cm_Soil_Temp_Min
20cm_Soil_Temp_Min_Time
50cm_Soil_Temp_Min
50cm_Soil_Temp_Min_Time
10cm_Soil_Temp_Max
10cm_Soil_Temp_Max_Time
20cm_Soil_Temp_Max
20cm_Soil_Temp_Max_Time
50cm_Soil_Temp_Max
50cm_Soil_Temp_Max_Time -------- Soil Temperatures at specified depth

Ground_Temp
Ground_Temp_Min
Ground_Temp_Min_Time
Ground_Temp_Max
Ground_Temp_Max_Time ----------- Surface Temperatures 

Rel_Humidity
Rel_Humidity_Min
Rel_Humidity_Min_Time
Rel_Humidity_Max
Rel_Humidity_Max_Time ---------- Relative Humidity (%)

10cm_Soil_Moisture
20cm_Soil_Moisture
50cm_Soil_Moisture
10cm_Soil_Moisture_Min
10cm_Soil_Moisture_Min_Time
20cm_Soil_Moisture_Min
20cm_Soil_Moisture_Min_Time
50cm_Soil_Moisture_Min
50cm_Soil_Moisture_Min_Time 
10cm_Soil_Moisture_Max
10cm_Soil_Moisture_Max_Time
20cm_Soil_Moisture_Max
20cm_Soil_Moisture_Max_Time
50cm_Soil_Moisture_Max
50cm_Soil_Moisture_Max_Time --- Soil Moisture Readings

Wind_Speed
Wind_Dir
Wind_Speed_Min
Wind_Speed_Min_Time
Wind_Speed_Max		
Wind_Speed_Max_Time ----------- Wind readings

Rain -------------------------- Rain Guage

Battery_V --------------------- Battery Voltage (mostly useless))


*************************************************************************************************************************
--------------------------------------------- HOURLY VARIABLE NAMES -----------------------------------------------------
*************************************************************************************************************************

Output_Code
Year
Day ---------------------------- These are largely unused, just a product of the datalogger at Cass and Chilton

Time --------------------------- Time in 'YYYY-mm-dd HH:MM:SS' formatting

Li190
Li200
Li190_Max
Li200_Max ---------------------- Li190 and Li200 are solar sensors for photolysis-active and total radiance respectively.

Air_Temp
Air_Temp_Min
Air_temp_Max ------------------- Mean, min and max air temperature readings

5m_Air_Temp
12m_Air_Temp ------------------- Air temperature at different heights (Chilton

Soil_Temp
10cm_Soil_Temp
20cm_Soil_Temp
50cm_Soil_Temp
10cm_Soil_Temp_Min
20cm_Soil_Temp_Min
50cm_Soil_Temp_Min
10cm_Soil_Temp_Max
20cm_Soil_Temp_Max
50cm_Soil_Temp_Max ------------- Soil Temperatures at depth (if specified)

Ground_Temp
Ground_Temp_Min
Ground_Temp_Max ---------------- Surface Temperatures 

Rel_Humidity
Rel_Humidity_Min
Rel_Humidity_Max --------------- Relative Humidity (%)

10cm_Soil_Moisture
20cm_Soil_Moisture
50cm_Soil_Moisture
10cm_Soil_Moisture_Min
20cm_Soil_Moisture_Min
50cm_Soil_Moisture_Min
10cm_Soil_Moisture_Max
20cm_Soil_Moisture_Max
50cm_Soil_Moisture_Max  -------- Soil Moisture Readings

Wind_Speed
Wind_Dir
Wind_Speed_Min
Wind_Speed_Max
Windrun
Gust_Speed
Gust_Dir ----------------------- Wind speed, direction, windrun, and gust readings

Wind_Speed_2
Wind_Dir_2
Wind_Dir_STD ------------------- A Secondary set of wind readings (Cass only)

Rain --------------------------- Rain Guage

BatVolt ------------------------ Still useless

Air_Temp_10Day
Rel_Humidity_10Day
Wind_Speed_10Day
Wind_Dir_10Day ----------------- 10 day summaries (Cass only)

Dew_Point ---------------------- Dew Point 

Wet_Bulb_Temp
Wet_Bulb_Temp_Min
Wet_Bulb_Temp_Max -------------- Wet-bulb readings
