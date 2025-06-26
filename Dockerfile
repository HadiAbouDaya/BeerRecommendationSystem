FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       build-essential \
       supervisor \
 && rm -rf /var/lib/apt/lists/*

# copy just the two requirements
COPY requirements-backend.txt requirements-frontend.txt ./

# create isolated venvs
RUN python -m venv /opt/venv-backend \
 && python -m venv /opt/venv-frontend

# install each
RUN /opt/venv-backend/bin/pip install --no-cache-dir -r requirements-backend.txt \
 && /opt/venv-frontend/bin/pip install --no-cache-dir -r requirements-frontend.txt

# copy your code
COPY . .

# supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
