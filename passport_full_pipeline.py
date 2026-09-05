"""
Passport Screening Pipeline — Module 1 + Module 2 + Module 3
--------------------------------------------------------------
AI-Based Fake Identity & Document Screening System

This combines:
    MODULE 1 (OCR Extraction)      — reads MRZ fields via PassportEye
    MODULE 2 (Document Validation) — MRZ checksums + document logic
                                      (expiry, age, country code, etc.)
    MODULE 3 (Tampering Detection) — image-level forgery detection:
                                      Error Level Analysis (ELA),
                                      Copy-Move detection, Metadata analysis

Requirements (all free):
    pip install passporteye opencv-python pillow pytesseract numpy

Also requires Tesseract OCR installed on your system (the actual program,
not just the Python wrapper):
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    Mac     : brew install tesseract
    Linux   : sudo apt install tesseract-ocr

HOW TO USE:
    Set IMAGE_PATH below, then run this file directly.
    No terminal arguments needed.
"""

import json
import re
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageChops
import cv2
from passporteye import read_mrz
import pytesseract

# Point pytesseract directly at the Tesseract executable (Windows fix)
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux (Render), pytesseract finds tesseract automatically via PATH — no override needed

# pytesseract is only needed for the optional visible-name cross-check
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


# ============================================================
# SETTINGS YOU CONTROL
# ============================================================
IMAGE_PATH = "C:\\Users\\DHEEPAESH VA\\OneDrive\\Desktop\\sih\\passportSample.jpg"
# Examples:
#   Same folder as this script :  "sample_passport.jpg"
#   Windows full path          :  r"C:\Users\YourName\Desktop\passport.jpg"
#   Mac/Linux full path        :  "/Users/YourName/Desktop/passport.jpg"

# OPTIONAL: cropped image of just the VISIBLE printed name on the photo
# page, for cross-checking against the MRZ name. Leave None to skip.
VISIBLE_NAME_REGION_PATH = None

# OPTIONAL: restrict Module 3's ELA + copy-move checks to a specific box
# (x, y, width, height) in pixels — e.g. the photo + text area, excluding
# decorative security patterns near the borders that can cause false
# positives. Leave None to analyze the whole image.
# NOTE: this was left as None before, which meant ELA/copy-move analyzed
# the ENTIRE image, including the busy security-pattern borders that caused
# the earlier false-positive (2012 matches on a genuine passport). Set this
# to a box around just the photo + printed text area on YOUR sample image
# (excluding the decorative patterned border) — the example values below
# are a placeholder and will need adjusting to your actual image's layout.
ANALYSIS_REGION = (40, 60, 450, 320)
# Example: ANALYSIS_REGION = (40, 60, 450, 320)

# OPTIONAL: bounding box of just the FACE PHOTO region on the document,
# as (x, y, width, height) in pixels. Needed only for the physical
# consistency check (photo-swap detection). Leave None to skip that check.
PHOTO_REGION = None
# Example: PHOTO_REGION = (60, 90, 140, 170)

# ---- Module 1/2 reference data ----
VALID_DOCUMENT_TYPES = ["P", "V", "I", "A", "C"]
VALID_COUNTRY_CODES = {
    "USA", "GBR", "IND", "CAN", "AUS", "DEU", "FRA", "JPN", "CHN", "BRA",
    "ZAF", "NGA", "PAK", "BGD", "RUS", "ITA", "ESP", "MEX", "IDN", "PHL",
    "UTO",
}
MAX_PLAUSIBLE_AGE = 110

# ---- Module 3 tunable thresholds ----
ELA_JPEG_QUALITY = 90
ELA_SCALE_FACTOR = 15
ELA_SUSPICION_THRESHOLD = 45

# These were raised drastically (60/500) to kill a false positive on a
# genuine passport's security-pattern border — but that overcorrected and
# made real, localized tampering (a few dozen matches) undetectable too.
# Now that ANALYSIS_REGION above restricts detection to just the photo +
# text area (excluding that noisy border), these can go back down to
# values sensitive enough to actually catch real tampering.
COPY_MOVE_MIN_DISTANCE_PX = 40
COPY_MOVE_MIN_MATCHES = 15

METADATA_SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "paint.net", "pixelmator", "affinity photo",
]

# ---- Physical consistency check thresholds (photo-swap detection) ----
# These compare the FACE PHOTO region against the surrounding document —
# designed to catch a PHYSICALLY swapped photo that was then photographed
# as a whole (a case ELA/copy-move generally can't see, since there's no
# digital edit history in a fresh camera JPEG).
NOISE_RATIO_SUSPICION_THRESHOLD = 1.8   # photo-region noise vs surrounding, ratio
LIGHTING_ANGLE_SUSPICION_DEGREES = 35   # gradient-direction mismatch, degrees

# ---- Overall combined risk weighting ----
# How much each module contributes to the FINAL combined risk score.
# NOTE: these must sum to 1.0 — the previous values (0.5 + 0.55 + 0.5 = 1.55)
# broke the 0-100 scale of the final score entirely.
WEIGHT_MRZ_LOGIC = 0.3   # Module 1/2's checksum + logic risk
WEIGHT_TAMPERING = 0.4   # Module 3's digital tampering score (ELA/copy-move/metadata)
WEIGHT_PHYSICAL = 0.3    # Module 3's physical consistency score (photo-swap check)
# ============================================================


# ============================================================
# MODULE 1: OCR EXTRACTION (PassportEye + Tesseract)
# ============================================================
def extract_mrz_data(image_path: str) -> dict:
    """
    Runs PassportEye on the given image and returns a structured
    dictionary of extracted + checksum-validated MRZ fields.
    """
    mrz = read_mrz(image_path)

    if mrz is None:
        return {
            "success": False,
            "error": "No MRZ detected in the image. Check image quality, "
                     "cropping, or lighting.",
        }

    data = mrz.to_dict()

    def format_mrz_date(raw_date):
        try:
            return datetime.strptime(raw_date, "%y%m%d").strftime("%d-%m-%Y")
        except (ValueError, TypeError):
            return None

    raw_text = data.get("raw_text", "") or ""
    raw_lines = raw_text.split("\n")

    return {
        "success": True,
        "raw_mrz_lines": {
            "line1": raw_lines[0] if len(raw_lines) > 0 else None,
            "line2": raw_lines[1] if len(raw_lines) > 1 else None,
        },
        "extracted_fields": {
            "document_type": data.get("type"),
            "issuing_country": data.get("country"),
            "surname": data.get("surname"),
            "given_names": data.get("names"),
            "passport_number": data.get("number"),
            "nationality": data.get("nationality"),
            "date_of_birth": format_mrz_date(data.get("date_of_birth")),
            "sex": data.get("sex"),
            "date_of_expiry": format_mrz_date(data.get("expiration_date")),
            "personal_number": data.get("personal_number"),
        },
        "checksum_validation": {
            "passport_number_valid": data.get("valid_number"),
            "date_of_birth_valid": data.get("valid_date_of_birth"),
            "expiry_date_valid": data.get("valid_expiration_date"),
            "personal_number_valid": data.get("valid_personal_number"),
            "composite_check_valid": data.get("valid_composite"),
            "overall_mrz_confidence": data.get("valid_score"),
        },
    }


def extract_visible_name(image_path: str):
    """Direct Tesseract OCR (not via PassportEye) on a cropped visible-name image."""
    if not PYTESSERACT_AVAILABLE or not image_path:
        return None
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, config="--psm 6")
        cleaned = re.sub(r"[^A-Za-z\s]", "", text).strip().upper()
        return cleaned if cleaned else None
    except Exception:
        return None


# ============================================================
# MODULE 2: DOCUMENT VALIDATION (checksum + logic checks)
# ============================================================
def compute_mrz_risk(result: dict, visible_name_path: str = None) -> dict:
    """
    Converts MRZ checksum validation + document-logic checks into risk
    flags and a combined risk contribution score (0 = clean, 100 = highly
    suspicious).
    """
    if not result.get("success"):
        return {"risk_contribution": 100, "flags": ["MRZ not readable at all"]}

    checks = result["checksum_validation"]
    fields = result["extracted_fields"]
    flags = []
    penalty = 0

    if checks.get("passport_number_valid") is False:
        flags.append("Passport number checksum failed — possible tampering")
        penalty += 30
    if checks.get("date_of_birth_valid") is False:
        flags.append("Date of birth checksum failed — possible tampering")
        penalty += 30
    if checks.get("expiry_date_valid") is False:
        flags.append("Expiry date checksum failed — possible tampering")
        penalty += 25
    if checks.get("composite_check_valid") is False:
        flags.append("Composite MRZ checksum failed — document likely altered")
        penalty += 40
    confidence = checks.get("overall_mrz_confidence")
    if confidence is not None and confidence < 70:
        flags.append(f"Low overall MRZ OCR confidence ({confidence})")
        penalty += (70 - confidence) * 0.5

    doc_type = fields.get("document_type")
    if doc_type and doc_type not in VALID_DOCUMENT_TYPES:
        flags.append(f"Unrecognized document type code: '{doc_type}'")
        penalty += 20

    expiry_raw = fields.get("date_of_expiry")
    if expiry_raw:
        try:
            expiry_dt = datetime.strptime(expiry_raw, "%d-%m-%Y")
            if expiry_dt < datetime.now():
                flags.append(f"Document expired on {expiry_raw}")
                penalty += 35
        except ValueError:
            pass

    issuing_country = fields.get("issuing_country")
    nationality = fields.get("nationality")
    for label, code in [("Issuing country", issuing_country),
                         ("Nationality", nationality)]:
        if code and code not in VALID_COUNTRY_CODES:
            flags.append(f"{label} code '{code}' not recognized/invalid")
            penalty += 15

    dob_raw = fields.get("date_of_birth")
    if dob_raw:
        try:
            dob_dt = datetime.strptime(dob_raw, "%d-%m-%Y")
            age_years = (datetime.now() - dob_dt).days / 365.25
            if age_years < 0:
                flags.append("Date of birth is in the future — invalid")
                penalty += 40
            elif age_years > MAX_PLAUSIBLE_AGE:
                flags.append(f"Implausible age ({age_years:.0f} years) based on DOB")
                penalty += 30
        except ValueError:
            pass

    if visible_name_path:
        visible_name = extract_visible_name(visible_name_path)
        mrz_name = f"{fields.get('surname', '')} {fields.get('given_names', '')}"
        mrz_name_clean = re.sub(r"[^A-Za-z\s]", "", mrz_name).strip().upper()
        mrz_name_clean = re.sub(r"\s+", " ", mrz_name_clean)

        if visible_name is None:
            flags.append("Visible name region unreadable — cross-check skipped")
        else:
            visible_name_clean = re.sub(r"\s+", " ", visible_name).strip()
            mrz_tokens = set(mrz_name_clean.split())
            visible_tokens = set(visible_name_clean.split())
            overlap = mrz_tokens & visible_tokens
            if mrz_tokens and len(overlap) < len(mrz_tokens) * 0.5:
                flags.append(
                    f"Visible name ('{visible_name_clean}') does not match "
                    f"MRZ name ('{mrz_name_clean}') — possible photo page tampering"
                )
                penalty += 35

    penalty = min(penalty, 100)
    return {
        "risk_contribution": round(penalty, 2),
        "flags": flags if flags else ["No checksum or logic anomalies detected"],
    }


# ============================================================
# MODULE 3: TAMPERING DETECTION (image-level)
# ============================================================
def crop_to_region(image, region):
    if region is None:
        return image
    x, y, w, h = region
    if isinstance(image, Image.Image):
        return image.crop((x, y, x + w, y + h))
    return image[y:y + h, x:x + w]


def run_ela(image_path: str, quality: int = ELA_JPEG_QUALITY,
            scale: int = ELA_SCALE_FACTOR, region=None):
    """
    Re-saves the image at a known JPEG quality, compares it pixel-by-pixel
    against the original, and amplifies the difference. Edited/pasted
    regions show up brighter than untouched regions.
    """
    original = Image.open(image_path).convert("RGB")
    original = crop_to_region(original, region)

    temp_path = image_path.rsplit(".", 1)[0] + "_ela_temp.jpg"
    original.save(temp_path, "JPEG", quality=quality)
    recompressed = Image.open(temp_path)

    diff = ImageChops.difference(original, recompressed)
    diff_np = np.array(diff).astype(np.float32) * scale
    diff_np = np.clip(diff_np, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(diff_np)

    gray = np.array(ela_image.convert("L")).astype(np.float32)
    mean_brightness = float(gray.mean())
    max_brightness = float(gray.max())

    os.remove(temp_path)
    return ela_image, mean_brightness, max_brightness


def evaluate_ela(image_path: str, region=None) -> dict:
    ela_image, mean_b, max_b = run_ela(image_path, region=region)

    output_path = image_path.rsplit(".", 1)[0] + "_ela.jpg"
    ela_image.save(output_path, "JPEG")

    suspicious = mean_b > ELA_SUSPICION_THRESHOLD
    return {
        "ela_heatmap_path": output_path,
        "mean_brightness": round(mean_b, 2),
        "max_brightness": round(max_b, 2),
        "suspicious": suspicious,
        "note": (
            f"Mean ELA brightness ({mean_b:.1f}) exceeds threshold "
            f"({ELA_SUSPICION_THRESHOLD}) — possible edited/pasted region"
            if suspicious else
            "ELA brightness within normal range"
        ),
    }


def detect_copy_move(image_path: str, region=None) -> dict:
    """
    ORB keypoint detection + brute-force matching to find near-identical
    regions within the SAME image that are spatially far apart — a sign
    part of the image was copy-pasted over another part.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"suspicious": False, "num_matches": 0,
                "note": "Could not load image for copy-move analysis"}

    img = crop_to_region(img, region)

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 10:
        return {"suspicious": False, "num_matches": 0,
                "note": "Not enough distinct features found for copy-move analysis"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    suspicious_matches = []
    for match_group in matches:
        for m in match_group[1:]:  # skip index 0 = self-match
            if m.distance < 20:
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                spatial_dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                if spatial_dist > COPY_MOVE_MIN_DISTANCE_PX:
                    suspicious_matches.append((pt1, pt2, float(m.distance)))

    num_matches = len(suspicious_matches)
    suspicious = num_matches >= COPY_MOVE_MIN_MATCHES
    return {
        "suspicious": suspicious,
        "num_matches": num_matches,
        "sample_matches": suspicious_matches[:5],
        "note": (
            f"Found {num_matches} spatially-separated near-duplicate regions "
            f"— possible copy-move forgery"
            if suspicious else
            f"Only {num_matches} weak duplicate signals — likely not copy-move forgery"
        ),
    }


def analyze_metadata(image_path: str) -> dict:
    """Reads EXIF metadata and flags editing-software traces or timestamp mismatches."""
    flags = []
    exif_data = {}

    try:
        img = Image.open(image_path)
        raw_exif = img._getexif()
        if raw_exif:
            from PIL.ExifTags import TAGS
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[str(tag_name)] = str(value)
    except Exception:
        pass

    if not exif_data:
        flags.append(
            "No EXIF metadata found — could indicate a screenshot, "
            "heavily processed image, or stripped metadata (weak signal on its own)"
        )
    else:
        software = exif_data.get("Software", "").lower()
        for tool in METADATA_SUSPICIOUS_SOFTWARE:
            if tool in software:
                flags.append(f"Image metadata shows editing software used: '{exif_data.get('Software')}'")
                break

        date_original = exif_data.get("DateTimeOriginal")
        date_modified = exif_data.get("DateTime")
        if date_original and date_modified and date_original != date_modified:
            flags.append(
                f"Creation timestamp ({date_original}) differs from modified "
                f"timestamp ({date_modified}) — image was edited after capture"
            )

    return {
        "exif_present": bool(exif_data),
        "flags": flags if flags else ["No suspicious metadata signals found"],
        "raw_fields_found": list(exif_data.keys()),
    }


def compute_module3_score(image_path: str, region=None) -> dict:
    """Combines ELA + copy-move + metadata into one image-tampering score."""
    ela_result = evaluate_ela(image_path, region=region)
    copy_move_result = detect_copy_move(image_path, region=region)
    metadata_result = analyze_metadata(image_path)

    flags = []
    penalty = 0

    if ela_result["suspicious"]:
        flags.append(ela_result["note"])
        excess = ela_result["mean_brightness"] - ELA_SUSPICION_THRESHOLD
        penalty += min(40, 20 + excess * 0.5)

    if copy_move_result["suspicious"]:
        flags.append(copy_move_result["note"])
        penalty += min(40, 15 + copy_move_result["num_matches"])

    real_metadata_flags = [
        f for f in metadata_result["flags"]
        if f != "No suspicious metadata signals found"
    ]
    if real_metadata_flags:
        flags.extend(real_metadata_flags)
        penalty += min(15, len(real_metadata_flags) * 8)

    penalty = min(round(penalty, 2), 100)
    return {
        "tampering_score": penalty,
        "flags": flags if flags else ["No tampering indicators detected"],
        "details": {"ela": ela_result, "copy_move": copy_move_result, "metadata": metadata_result},
    }


# ============================================================
# MODULE 3 (BONUS): PHYSICAL CONSISTENCY CHECK — PHOTO-SWAP DETECTION
# ============================================================
# ELA and copy-move detection catch DIGITAL edits made to the image
# file itself. They do NOT catch a physically swapped photo on a real
# passport that is then photographed as a whole — that photo has no
# "digital edit history," since it was compressed only once, at the
# moment of capture.
#
# This check instead compares the FACE PHOTO region against the
# surrounding document using two signals that a physical photo swap
# tends to disturb, even within a single, unedited camera shot:
#
#   1. Noise/texture consistency — a printed photo, especially one
#      swapped in from a different source/printer, usually has a
#      different fine-grain noise pattern than the surrounding
#      document, since it may have been printed on different paper,
#      with a different printer, or re-photographed from a screen.
#
#   2. Lighting/gradient direction consistency — under a single light
#      source, brightness gradients across a flat document surface
#      should point in roughly the same direction everywhere. A
#      swapped photo — sitting slightly proud of the surface, at a
#      faint angle, or under a laminate with different reflectivity —
#      often shows a noticeably different gradient direction than the
#      surrounding page.
#
# Both signals are WEAKER than ELA/copy-move on their own (lighting and
# printer noise vary naturally even in genuine documents), so this is
# deliberately given a smaller weight in the final combined score.
# ------------------------------------------------------------
def compute_noise_consistency(image_path: str, photo_region) -> dict:
    """
    Compares local noise/texture variance inside the photo region against
    the surrounding document area. A big mismatch suggests the photo
    region came from a different physical source than the rest of the
    page (e.g. a different printer, paper, or a re-photographed screen).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"suspicious": False, "note": "Could not load image"}

    x, y, w, h = photo_region
    photo_crop = img[y:y + h, x:x + w]

    # "Surrounding" = whole image with the photo region blanked out,
    # so we're comparing texture against the rest of the document only.
    surrounding = img.copy()
    surrounding[y:y + h, x:x + w] = 0

    # Laplacian variance is a standard proxy for high-frequency texture/
    # noise: a sharper, higher-frequency printed surface produces a
    # higher variance; a smoother/flatter surface produces a lower one.
    photo_noise = cv2.Laplacian(photo_crop, cv2.CV_64F).var()
    surrounding_noise = cv2.Laplacian(surrounding, cv2.CV_64F).var()

    if surrounding_noise == 0:
        ratio = 1.0
    else:
        ratio = photo_noise / surrounding_noise

    suspicious = (ratio > NOISE_RATIO_SUSPICION_THRESHOLD) or (
        ratio < 1 / NOISE_RATIO_SUSPICION_THRESHOLD
    )

    return {
        "suspicious": suspicious,
        "photo_region_noise": round(float(photo_noise), 2),
        "surrounding_noise": round(float(surrounding_noise), 2),
        "noise_ratio": round(float(ratio), 2),
        "note": (
            f"Photo region texture differs sharply from surrounding document "
            f"(ratio {ratio:.2f}) — possible photo swap"
            if suspicious else
            "Photo region texture consistent with surrounding document"
        ),
    }


def compute_lighting_consistency(image_path: str, photo_region) -> dict:
    """
    Compares the average brightness-gradient DIRECTION inside the photo
    region against the surrounding document, using Sobel gradients. A
    genuine single photo of a flat document should show a fairly
    consistent lighting direction everywhere; a swapped photo sitting at
    a slightly different angle/reflectivity often breaks that
    consistency.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"suspicious": False, "note": "Could not load image"}

    x, y, w, h = photo_region
    photo_crop = img[y:y + h, x:x + w].astype(np.float32)

    surrounding = img.copy().astype(np.float32)
    surrounding[y:y + h, x:x + w] = np.nan  # mark photo area to exclude from mean

    def mean_gradient_angle(patch):
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        # Only consider pixels with a meaningfully strong gradient —
        # flat/uniform areas have near-random, noisy gradient directions
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        mask = magnitude > np.nanpercentile(magnitude, 75)
        if not np.any(mask):
            return 0.0
        angles = np.arctan2(gy[mask], gx[mask])
        mean_angle = np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles)))
        return np.degrees(mean_angle)

    photo_angle = mean_gradient_angle(photo_crop)

    # For the surrounding area, compute gradients ignoring the NaN hole
    surrounding_filled = np.nan_to_num(surrounding, nan=float(np.nanmean(surrounding)))
    surrounding_angle = mean_gradient_angle(surrounding_filled)

    angle_diff = abs(photo_angle - surrounding_angle)
    angle_diff = min(angle_diff, 360 - angle_diff)  # wrap-around

    suspicious = angle_diff > LIGHTING_ANGLE_SUSPICION_DEGREES

    return {
        "suspicious": suspicious,
        "photo_region_angle_deg": round(float(photo_angle), 1),
        "surrounding_angle_deg": round(float(surrounding_angle), 1),
        "angle_difference_deg": round(float(angle_diff), 1),
        "note": (
            f"Lighting gradient direction in photo region differs by "
            f"{angle_diff:.0f}\u00b0 from surrounding document — possible photo swap"
            if suspicious else
            "Lighting direction in photo region consistent with surrounding document"
        ),
    }


def compute_physical_consistency_score(image_path: str, photo_region=None) -> dict:
    """
    Combines noise consistency + lighting consistency into one physical
    tampering score. Returns a score of 0 (no photo_region given, check
    skipped) up to 100 (strong signs of a physically swapped photo).
    """
    if not photo_region:
        return {
            "physical_score": 0,
            "flags": ["PHOTO_REGION not set — physical consistency check skipped"],
            "details": None,
        }

    noise_result = compute_noise_consistency(image_path, photo_region)
    lighting_result = compute_lighting_consistency(image_path, photo_region)

    flags = []
    penalty = 0

    if noise_result["suspicious"]:
        flags.append(noise_result["note"])
        penalty += 60

    if lighting_result["suspicious"]:
        flags.append(lighting_result["note"])
        penalty += 60

    # BUG FIX: this was `max(penalty, 200)`, which forced physical_score to
    # ALWAYS be at least 200 whenever PHOTO_REGION was set — regardless of
    # whether anything suspicious was actually found. Correct behavior is
    # to cap the score at 100, not force a minimum of 200.
    penalty = min(penalty, 100)

    return {
        "physical_score": penalty,
        "flags": flags if flags else ["No physical photo-swap indicators detected"],
        "details": {"noise": noise_result, "lighting": lighting_result},
    }


# ============================================================
# COMBINE ALL MODULES INTO ONE FINAL RISK SCORE
# ============================================================
def run_full_pipeline(image_path: str, visible_name_path: str = None,
                       analysis_region=None, photo_region=None) -> dict:
    extraction = extract_mrz_data(image_path)
    mrz_risk = compute_mrz_risk(extraction, visible_name_path)
    tampering = compute_module3_score(image_path, region=analysis_region)
    physical = compute_physical_consistency_score(image_path, photo_region=photo_region)

    # If the physical consistency check was SKIPPED (no PHOTO_REGION set),
    # its score of 0 means "not checked," not "no risk found" — including
    # it at full weight would unfairly drag the final score down. Instead,
    # redistribute its weight proportionally across the checks that
    # actually ran, so a skipped check has no effect on the final score
    # either way.
    physical_was_skipped = physical["details"] is None

    if physical_was_skipped:
        remaining_weight = WEIGHT_MRZ_LOGIC + WEIGHT_TAMPERING
        w_mrz = WEIGHT_MRZ_LOGIC / remaining_weight
        w_tampering = WEIGHT_TAMPERING / remaining_weight
        w_physical = 0
    else:
        w_mrz = WEIGHT_MRZ_LOGIC
        w_tampering = WEIGHT_TAMPERING
        w_physical = WEIGHT_PHYSICAL

    final_score = round(
        mrz_risk["risk_contribution"] * w_mrz
        + tampering["tampering_score"] * w_tampering
        + physical["physical_score"] * w_physical,
        2,
    )

    return {
        "final_risk_score": final_score,
        "module1_extraction": extraction,
        "module2_mrz_risk": mrz_risk,
        "module3_tampering": tampering,
        "module3_physical_consistency": physical,
    }


def main():
    print(f"\nProcessing: {IMAGE_PATH}\n{'=' * 60}")

    result = run_full_pipeline(
        IMAGE_PATH,
        visible_name_path=VISIBLE_NAME_REGION_PATH,
        analysis_region=ANALYSIS_REGION,
        photo_region=PHOTO_REGION,
    )

    extraction = result["module1_extraction"]
    if extraction["success"]:
        print("\nMODULE 1 — EXTRACTED FIELDS:")
        for k, v in extraction["extracted_fields"].items():
            print(f"  {k:20s}: {v}")
    else:
        print(f"\nMODULE 1 FAILED: {extraction['error']}")

    print("\nMODULE 2 — MRZ / LOGIC RISK:")
    print(f"  Risk contribution: {result['module2_mrz_risk']['risk_contribution']} / 100")
    for f in result["module2_mrz_risk"]["flags"]:
        print(f"  - {f}")

    print("\nMODULE 3 — IMAGE TAMPERING RISK (digital edits):")
    print(f"  Tampering score: {result['module3_tampering']['tampering_score']} / 100")
    for f in result["module3_tampering"]["flags"]:
        print(f"  - {f}")

    print("\nMODULE 3 (BONUS) — PHYSICAL CONSISTENCY (photo-swap check):")
    print(f"  Physical score: {result['module3_physical_consistency']['physical_score']} / 100")
    for f in result["module3_physical_consistency"]["flags"]:
        print(f"  - {f}")

    print(f"\n{'=' * 60}")
    print(f"FINAL COMBINED RISK SCORE: {result['final_risk_score']} / 100")
    print(f"{'=' * 60}")

    output_path = IMAGE_PATH.rsplit(".", 1)[0] + "_full_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull result saved to: {output_path}")


if __name__ == "__main__":
    main()
