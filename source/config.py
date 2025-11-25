import os

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'

# === Dropbox Sync Requirements (DBX) ===
LOG_DBX         = True
SAMPLE_RATE_DBX = 0.01
DRY_RUN_DBX     = False
OUTPUT_DBX      = False
FLAT_DBX        = True
EXCLUDE_DBX     = []
WANT_DBX        = ["sps"]
DESTINATION_PATH_DBX = "./_sps_data"
# =================================

# === SPS Converter (STN) ===
SPS_DUMP_STN =  DESTINATION_PATH_DBX
NPY_DIR_STN  = "./_raw_npy"
SHOW_SPS_STN = False
MAKE_NPY_STN = True
MAKE_CSV_STN = False
COMPRESS_STN = False
# =================================

# === Preprocessing Data (PRP) ===
NPY_DUMP_PRP    = NPY_DIR_STN
FILL_ZERO_PRP   = True
PREPROCESS_PRP  = False
CLEANED_NPY_PRP = "./_cleaned_npy"
# =================================

# ======== SRB Detection ========

# === Radon Analysis (RA) ===
INPUT_DIR_RA    =  CLEANED_NPY_PRP
OUTPUT_DIR_RA   = "./_detected_srb"
CREATE_IMG_RA   = True
# ============================

if __name__ == "__main__":
    # Check all file paths exist
    if not os.path.exists(DESTINATION_PATH_DBX):
        os.makedirs(DESTINATION_PATH_DBX)
    if not os.path.exists(NPY_DIR_STN):
        os.makedirs(NPY_DIR_STN)
    if not os.path.exists(CLEANED_NPY_PRP):
        os.makedirs(CLEANED_NPY_PRP)
    if not os.path.exists(OUTPUT_DIR_RA):
        os.makedirs(OUTPUT_DIR_RA)

    print(f"{GREEN}All output folders exist! {RESET}")