explicame esto:
DATA_PATH = Path("data/daily_fitbit_sema_df_unprocessed.csv")

COL_ID = "id"
RANDOM_STATE = 42
N_FOLDS = 5          # Stratified K-Fold (n=71 -> ~14 personas por fold)
N_REPEATS = 20       # se repite el 5-Fold con 20 semillas distintas y se promedia
RISK_LABEL_QUANTILE = 0.25  # bottom 25% de calidad de sueño = "riesgo alto"

FEATURE_COLUMNS_PHYSIO = [
    "rmssd", "nremhr", "resting_hr", "spo2", "full_sleep_breathing_rate",
    "minutesToFallAsleep", "minutesAsleep", "minutesAwake",
    "sleep_efficiency", "sleep_points_percentage",
]
FEATURE_COLUMNS_BEHAVIORAL = ["steps", "sedentary_minutes"]
ALL_RAW_COLUMNS = FEATURE_COLUMNS_PHYSIO + FEATURE_COLUMNS_BEHAVIORAL


quiero saber las variables predictoras.