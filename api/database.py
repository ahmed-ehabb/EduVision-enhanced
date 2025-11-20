"""
Database connection and session management for EduVision
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(dotenv_path="../database/.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://eduvision_app:eduvision_dev_password_2025@localhost:5432/eduvision"
)

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before using them
    echo=False,  # Set to True for SQL logging during development
    future=True
)

# Create sessionmaker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# ============================================================================
# Session Management
# ============================================================================

def get_db() -> Session:
    """
    Dependency function for FastAPI to get database session

    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database session

    Usage:
        with get_db_session() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================================
# Database Initialization
# ============================================================================

def init_db():
    """
    Initialize database - create all tables
    NOTE: Tables are already created by schema.sql,
    this is here for reference/testing
    """
    from models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")


def check_connection():
    """
    Test database connection
    Returns True if connection successful
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("[OK] Database connection successful")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Database connection failed: {e}")
        return False


def get_table_count():
    """
    Get count of tables in database
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            )
            count = result.scalar()
        logger.info(f"Found {count} tables in database")
        return count
    except Exception as e:
        logger.error(f"Failed to get table count: {e}")
        return 0


# ============================================================================
# Connection Pool Monitoring
# ============================================================================

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log when new connection is created"""
    logger.debug("New database connection established")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log when connection is checked out from pool"""
    logger.debug("Connection checked out from pool")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log when connection is returned to pool"""
    logger.debug("Connection returned to pool")


def get_pool_status():
    """
    Get current connection pool status
    Returns dict with pool statistics
    """
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow()
    }


# ============================================================================
# Utilities
# ============================================================================

def reset_database():
    """
    DANGER: Drop all tables and recreate
    Only use in development!
    """
    from models import Base
    logger.warning("⚠️ Dropping all tables!")
    Base.metadata.drop_all(bind=engine)
    logger.info("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database reset complete")


def close_all_connections():
    """
    Close all database connections
    Useful for cleanup or testing
    """
    engine.dispose()
    logger.info("All database connections closed")


# ============================================================================
# Startup Check
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EduVision Database Connection Test")
    print("=" * 70)

    # Test connection
    if check_connection():
        print(f"[OK] Connected to: {DATABASE_URL.split('@')[1]}")  # Hide password

        # Get table count
        table_count = get_table_count()
        print(f"[OK] Found {table_count} tables")

        # Pool status
        pool_status = get_pool_status()
        print(f"[OK] Connection pool size: {pool_status['size']}")
        print(f"   Active connections: {pool_status['checked_out']}")
        print(f"   Idle connections: {pool_status['checked_in']}")

        print("\n" + "=" * 70)
        print("Database is ready!")
        print("=" * 70)
    else:
        print("[ERROR] Database connection failed!")
        print("Check your .env file and ensure PostgreSQL is running")
