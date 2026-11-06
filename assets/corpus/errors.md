# Archive service error codes

ARC-17 means the archive checksum differs from the uploaded manifest. Quarantine the upload, recompute the checksum from the original file, and submit a corrected manifest before retrying.

ARC-42 means the account has reached its concurrent export limit. Wait for an active export to finish before starting another.
