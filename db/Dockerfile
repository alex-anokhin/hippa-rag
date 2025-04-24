FROM postgres:15

# Install PostgreSQL extensions
RUN apt-get update && apt-get install -y \
    postgresql-15-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Copy initialization scripts
# COPY init.sql /docker-entrypoint-initdb.d/