"""
database.py — Neon/Postgres engine and session factory.

Neon is the source of truth for accounts, chat sessions and messages.
Pinecone holds only vectors; a chat's identity and ownership live here.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set. Add your Neon connection string to backend/.env:\n"
        "  DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.region.aws.neon.tech/dbname?sslmode=require"
    )

# SQLAlchemy dropped the postgres:// scheme; Neon still hands it out in places.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Neon closes idle connections, which surfaces as "server closed the connection
# unexpectedly" on the next use. pool_pre_ping checks liveness before handing a
# connection out; pool_recycle retires them before Neon does.
# Use the -pooler host in DATABASE_URL so PgBouncer multiplexes these onto few
# real backends — otherwise a free-tier compute runs out of connections.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
