FROM ghcr.io/berriai/litellm:main-stable@sha256:9e1536c6a9219519f024f221706b20b012ca5176988164798adc5c7fe011e5d5

RUN apk add git
RUN pip install git+https://github.com/niozow/chutes-e2ee-transport.git@e0ccc6ef1d2c0a69c178a56b73dccb68bdaa4654 httpx pyyaml

COPY chutes_provider.py /app/chutes_provider.py
COPY generate_config.py /app/generate_config.py
COPY config.template.yml /app/config.template.yml

ENTRYPOINT ["sh", "-c", "python /app/generate_config.py && exec litellm --config /app/config.generated.yml --host 0.0.0.0 --port 4000"]
