import pandas as pd

# Load the CSV files
idm_2021 = pd.read_csv('Data/idm_baru - IDM 2021.csv')
idm_2022 = pd.read_csv('Data/idm_baru - IDM 2022.csv')

# Normalize the `desa` and `kecamatan` names by converting to lowercase and stripping extra spaces
idm_2021['DESA'] = idm_2021['DESA'].str.lower().str.strip()
idm_2021['KECAMATAN'] = idm_2021['KECAMATAN'].str.lower().str.strip()

idm_2022['DESA'] = idm_2022['DESA'].str.lower().str.strip()
idm_2022['KECAMATAN'] = idm_2022['KECAMATAN'].str.lower().str.strip()

# Create a mapping for the desa names with differences
desa_name_mapping = {
    'alue bade': 'alue badee',
    'guha ulue': 'guha uleu',
    'keude matang payang': 'keude matang panyang',
    'tgk.dibanda pirak': 'tgk. dibanda pirak',
    'ujong  kulam': 'ujong kulam'
}

# Apply the mapping to the 2022 dataset
idm_2022['DESA'] = idm_2022['DESA'].replace(desa_name_mapping)

# Merge the two datasets based on `desa` and `kecamatan`
merged_data = pd.merge(idm_2022, idm_2021[['DESA', 'KECAMATAN', 'KODE BPS']], on=['DESA', 'KECAMATAN'], how='left')

# Check for rows without a match (i.e., without KODE BPS)
no_match = merged_data[merged_data['KODE BPS'].isnull()]
print(f"Rows without a match: \n{no_match}")

# Save the updated dataframe
merged_data.to_csv('Data/idm_2022_with_kode_bps.csv', index=False)
