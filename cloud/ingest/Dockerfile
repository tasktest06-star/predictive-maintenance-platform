# Ingest service image.
#
# Build context is the repository root so that core/ can be installed: the
# ingest service and the firmware share one feature definition, and a container
# that vendors its own copy would be able to drift from it.
FROM python:3.12-slim AS base

# Non-root from the start. The service terminates untrusted input from the
# field, so it gets the smallest blast radius available for the effort.
RUN useradd --create-home --uid 10001 pdm

WORKDIR /app
COPY core/ /app/core/
# numpy+scipy only: the detectors that ship on day one need neither sklearn nor
# torch, and keeping them out holds the image near 200 MB instead of 2 GB.
RUN pip install --no-cache-dir "./core" "paho-mqtt>=1.6" \
    && python -c "import pdmcore, pdmcore.dsp, pdmcore.health, pdmcore.io; print(pdmcore.__version__)"

COPY cloud/ingest/app.py /app/app.py

USER pdm
EXPOSE 8080
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# No shell form: the service must receive SIGTERM directly so an orchestrator
# can drain it, and an in-flight frame is never worth a SIGKILL.
ENTRYPOINT ["python", "/app/app.py"]
CMD ["--mqtt"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')"
