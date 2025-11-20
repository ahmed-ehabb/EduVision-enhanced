# Quick Fix: PostgreSQL Not Running

## Problem
The login is failing because PostgreSQL database is not running.

Error: `connection to server at "localhost" port 5432 failed: Connection refused`

## Solution

### Option 1: Start PostgreSQL Service (Windows)

1. **Open Services** (Press `Win + R`, type `services.msc`, press Enter)
2. **Find PostgreSQL service** (might be named like "postgresql-x64-14" or similar)
3. **Right-click** → **Start**

OR use Command Prompt as Administrator:
```cmd
net start postgresql-x64-14
```
(Replace `postgresql-x64-14` with your actual service name)

### Option 2: Start PostgreSQL Manually

If you installed PostgreSQL with default settings:

1. Open Command Prompt
2. Navigate to PostgreSQL bin directory:
```cmd
cd "C:\Program Files\PostgreSQL\14\bin"
```
3. Start the server:
```cmd
pg_ctl -D "C:\Program Files\PostgreSQL\14\data" start
```

### Option 3: Check if PostgreSQL is Installed

```cmd
psql --version
```

If not installed, download from: https://www.postgresql.org/download/windows/

## Verify PostgreSQL is Running

```cmd
netstat -ano | findstr :5432
```

You should see something like:
```
TCP    0.0.0.0:5432    0.0.0.0:0    LISTENING    1234
```

## After Starting PostgreSQL

1. **Initialize the database** (first time only):
```cmd
cd api
python setup_db.py
```

2. **Restart the Unified API**:
   - The API should automatically reconnect once PostgreSQL is running
   - If not, restart it: Stop with Ctrl+C and run `python main.py` again

3. **Try logging in again** at http://localhost:3000

## Test Credentials

After database is initialized:
- Email: `teacher@example.com`
- Password: `password123`

## Still Having Issues?

Check the database connection string in `api/.env`:
```
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/eduvision
```

Make sure:
- Username is correct (default: `postgres`)
- Password matches your PostgreSQL installation
- Database name is `eduvision` (will be created by `setup_db.py`)
