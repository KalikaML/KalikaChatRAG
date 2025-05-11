# config.py

# Email Secrets will be loaded from st.secrets in other modules where needed.
# This file can hold non-sensitive configurations or defaults.

IMAP_SERVER_DEFAULT = "imap.gmail.com"
DB_HOST_DEFAULT = "localhost"
DB_NAME_DEFAULT = "po_orders"
DB_USER_DEFAULT = "po_user"
DB_PASSWORD_DEFAULT = "postdb123"
DB_PORT_DEFAULT = 5432

# Email Subjects to Monitor (can also be in secrets.toml)
DEFAULT_EMAIL_SUBJECTS_TO_MONITOR = [
    "PO released // Consumable items",
    "PO copy",
    "import po",
    "RFQ-Polybag",
    "PFA PO",
    "Purchase Order FOR",
    "Purchase Order_"
]

# Scheduler Defaults
DEFAULT_SCHEDULER_INTERVAL_MINUTES = 60
DEFAULT_SCHEDULER_AUTO_ENABLE = False
DEFAULT_SCHEDULER_RECENT_DAYS = 3 # For email checks by scheduler
DEFAULT_MANUAL_CHECK_RECENT_DAYS = 7
DEFAULT_PENDING_COUNT_CACHE_MIN = 1

# WhatsApp
DEFAULT_SELLER_TEAM_RECIPIENTS_STR = "9284238738"