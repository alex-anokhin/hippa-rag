from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine
from contextlib import asynccontextmanager

from app.config import DATABASE_URL

Base = declarative_base()

# Create async engine
engine: AsyncEngine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True
)

# Create async session
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def get_db_session():
    """Database session dependency"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise