import os
import json
import pandas as pd
from firebase_admin import credentials, firestore, initialize_app, get_app

def init_firebase():
    """Initializes Firebase app if credentials are found in Streamlit secrets or env vars."""
    try:
        # Check if already initialized
        get_app()
        db = firestore.client()
        return db
    except ValueError:
        pass # App not initialized yet

    try:
        import streamlit as st
        if "firebase" in st.secrets:
            # Reconstruct the service account json from secrets
            cert = dict(st.secrets["firebase"])
            # If private_key contains escaped newlines, replace them
            if 'private_key' in cert:
                cert['private_key'] = cert['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(cert)
            initialize_app(cred)
            return firestore.client()
    except Exception as e:
        pass

    try:
        # Fallback to environment variable containing JSON string
        firebase_cert_json = os.environ.get("FIREBASE_CREDENTIALS")
        if firebase_cert_json:
            cert = json.loads(firebase_cert_json)
            cred = credentials.Certificate(cert)
            initialize_app(cred)
            return firestore.client()
    except Exception as e:
        pass
        
    return None

def sync_dataframe_to_firestore(db, df, collection_name, key_cols):
    """Syncs a pandas dataframe to Firestore, using a compound key for document ID."""
    if db is None:
        return
        
    batch = db.batch()
    collection_ref = db.collection(collection_name)
    
    # Process in batches of 500 (Firestore limit)
    count = 0
    for _, row in df.iterrows():
        doc_id = "_".join([str(row[col]) for col in key_cols])
        doc_ref = collection_ref.document(doc_id)
        batch.set(doc_ref, row.to_dict(), merge=True)
        count += 1
        
        if count == 500:
            batch.commit()
            batch = db.batch()
            count = 0
            
    if count > 0:
        batch.commit()

def sync_predictions(db, df):
    """Sync the AI predictions to Firestore"""
    if db is None:
        return
    # Only sync a subset to avoid overwhelming Firestore in a hackathon
    subset = df[['phc_id', 'medicine', 'predicted_demand_7d', 'stockout_probability', 'risk_score', 'risk_level']].copy()
    sync_dataframe_to_firestore(db, subset, 'predictions', ['phc_id', 'medicine'])
