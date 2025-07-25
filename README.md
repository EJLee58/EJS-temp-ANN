# Real-time estimation for 3-D temperature
A code set for real-time generation of 3-D temperature fields in the East (Japan) Sea, bounded by 35°–44°N and 128°–140°E.
Daily remote sensing data is used to predict sea temperature for the same day, with a 1/12° spatial resolution (latitude × longitude).

## 💻 Usage  
Run **`Model_run.bat`**.  
The script automatically downloads OSTIA and AVISO data from Copernicus via the code source, processes the input data, and performs temperature prediction using the model.  
The prediction results are saved as NetCDF4 files in the **`01_3D_temp_results`** folder.  
**Note:** The downloaded and processed data for the execution date will be stored in `00_Data_n_wgts/02_Daily_data`.

## 🧾 Requirements
- `tensorflow`  
- `copernicusmarine`  
  *(Requires a Copernicus Marine account and login credentials: [https://marine.copernicus.eu](https://marine.copernicus.eu))*

📄 For development dependencies, see [`dev_environment.txt`](./dev_environment.txt).
