#!/bin/bash
# apt install -y tpm2-tools
# --- Configuration ---
# Target persistent handle for the new SRK (Commonly the first available slot)
PERSISTENT_HANDLE="0x81010001"

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
tpm2_createprimary -C o -g sha256 -G rsa -c o.ctx
tpm2_create -C o.ctx -g sha256 -G aes256cfb -u mysymmetrickey.pub -r mysymmetrickey.priv
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create the transient SRK."
    # Attempt to clean up even if creation failed
    rm -f o.ctx mysymmetrickey.pub mysymmetrickey.priv
    exit 1
fi
echo "✅ Transient SRK created and saved to o.ctx."

# --- 3. Load to the TPM ---
echo "--- Step 3: Load both the private and public portions of an object into the TPM ---"
tpm2_load -C o.ctx -u mysymmetrickey.pub -r mysymmetrickey.priv -c mysymmetrickey.ctx
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to load private and public to the TPM."
    # Attempt to clean up even if creation failed
    rm -f o.ctx mysymmetrickey.pub mysymmetrickey.priv
    exit 1
fi
echo "✅ Load successfully."

# --- 4. Persist the SRK ---
echo "--- Step 4: Making the SRK permanent (Persistent Handle: ${PERSISTENT_HANDLE}) ---"
# Move the key from the transient handle (in the context file) to the persistent handle
# The '-C o' provides authorization to move the key under the Owner Hierarchy
tpm2_evictcontrol -C o -c mysymmetrickey.ctx "${PERSISTENT_HANDLE}"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to persist the SRK. Check if the handle ${PERSISTENT_HANDLE} is already in use."
    # Clean up transient context file
    rm -f o.ctx mysymmetrickey.pub mysymmetrickey.priv
    exit 1
fi
echo "✅ SRK successfully persisted to handle ${PERSISTENT_HANDLE}."

# --- 5. Verification and Cleanup ---
echo "--- Step 5: Verification and Cleanup ---"

# Verify the new key is present by attempting to read its public part
echo "🔍 Verifying the key at persistent handle ${PERSISTENT_HANDLE}..."
tpm2_readpublic -c "${PERSISTENT_HANDLE}"
if [ $? -ne 0 ]; then
    echo "❌ CRITICAL ERROR: Could not read the persistent key. Persistence failed."
    exit 1
fi
echo "✅ New SRK successfully verified at persistent handle."

# Clean up the temporary context file
rm -f o.ctx mysymmetrickey.pub mysymmetrickey.priv
echo "🗑️ Cleaned up temporary context file o.ctx."

echo ""
echo "✨ Script complete. The TPM is reset, and a new persistent SRK is ready for use."