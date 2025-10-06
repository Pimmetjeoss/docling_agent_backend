"""
Database Utilities
==================
Manages the global database connection pool.
"""

import os
import logging
import asyncpg

logger = logging.getLogger(__name__)

# Global database pool, to be initialized on application startup
db_pool = None

async def initialize_db():
    """Initialize the global database connection pool."""
    global db_pool
    if not db_pool:
        try:
            db_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL"),
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}", exc_info=True)
            raise

async def close_db():
    """Close the global database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None
        logger.info("Database connection pool closed.")
