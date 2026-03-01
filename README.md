# ML API Starter — Breast Cancer Prediction (FastAPI + Docker)

A production-minded ML inference microservice that serves a trained scikit-learn classifier via a FastAPI API.  
Built with validation, logging, request timing middleware, and containerized for deployment.

## What it does
- Loads a pre-trained model at startup
- Accepts 30 numerical features and returns:
  - prediction (0/1)
  - probability for class 1
  - model version + environment metadata

> Dataset: Breast Cancer Wisconsin (Diagnostic) from scikit-learn.

---

## Live API
- Base URL: 
- Docs (Swagger): `https://ml-api-starter.onrender.com/docs`
- Health: `https://ml-api-starter.onrender.com/health`

---

## Endpoints

### `GET /health`
Quick service health check.

**Response**
```json
{ "status": "ok" }