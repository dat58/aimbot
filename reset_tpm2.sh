#!/bin/bash
# apt install -y tpm2-tools
# --- Configuration ---
# Target persistent handle for the new SRK (Commonly the first available slot)
PERSISTENT_HANDLE="0x81000000"
# Context file to temporarily hold the transient SRK handle
CONTEXT_FILE="primary.ctx"

echo "🔐 Starting TPM Clear, SRK Creation, and Persistence..."

# --- 1. Clear the TPM ---
# WARNING: This deletes all existing keys and ownership info (e.g., BitLocker keys)!
echo "--- Step 1: Clearing the TPM (Factory Reset) ---"
# Clearing the Lockout hierarchy effectively clears Owner and Endorsement hierarchies too.
tpm2_clear
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to clear the TPM. Check permissions or if a password is set."
    exit 1
fi
echo "✅ TPM cleared successfully."

# --- 2. Create the Transient SRK ---
echo "--- Step 2: Creating a new transient Storage Root Key (SRK) ---"
# Create an RSA 2048-bit primary key under the Owner Hierarchy (-C o)
tpm2_createprimary -C o -G rsa -c "${CONTEXT_FILE}"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create the transient SRK."
    # Attempt to clean up even if creation failed
    rm -f "${CONTEXT_FILE}"
    exit 1
fi
echo "✅ Transient SRK created and saved to ${CONTEXT_FILE}."

# --- 3. Persist the SRK ---
echo "--- Step 3: Making the SRK permanent (Persistent Handle: ${PERSISTENT_HANDLE}) ---"
# Move the key from the transient handle (in the context file) to the persistent handle
# The '-C o' provides authorization to move the key under the Owner Hierarchy
tpm2_evictcontrol -C o -c "${CONTEXT_FILE}" -H "${PERSISTENT_HANDLE}"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to persist the SRK. Check if the handle ${PERSISTENT_HANDLE} is already in use."
    # Clean up transient context file
    rm -f "${CONTEXT_FILE}"
    exit 1
fi
echo "✅ SRK successfully persisted to handle ${PERSISTENT_HANDLE}."

# --- 4. Verification and Cleanup ---
echo "--- Step 4: Verification and Cleanup ---"

# Verify the new key is present by attempting to read its public part
echo "🔍 Verifying the key at persistent handle ${PERSISTENT_HANDLE}..."
tpm2_readpublic -c "${PERSISTENT_HANDLE}"
if [ $? -ne 0 ]; then
    echo "❌ CRITICAL ERROR: Could not read the persistent key. Persistence failed."
    exit 1
fi
echo "✅ New SRK successfully verified at persistent handle."

# Clean up the temporary context file
rm -f "${CONTEXT_FILE}"
echo "🗑️ Cleaned up temporary context file ${CONTEXT_FILE}."

echo ""
echo "✨ Script complete. The TPM is reset, and a new persistent SRK is ready for use."