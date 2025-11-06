import os
import joblib
import pefile
import pandas as pd
import math

def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(bytes([x]))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_raw_features(file_path):
    try:
        pe = pefile.PE(file_path)
    except Exception as e:
        print(f"Error parsing PE file: {e}")
        return pd.DataFrame()  # Return empty if file invalid

    # Base header features
    features = {
        "Machine": pe.FILE_HEADER.Machine,
        "NumberOfSections": pe.FILE_HEADER.NumberOfSections,
        "TimeDateStamp": pe.FILE_HEADER.TimeDateStamp,
        "PointerToSymbolTable": pe.FILE_HEADER.PointerToSymbolTable,
        "NumberOfSymbols": pe.FILE_HEADER.NumberOfSymbols,
        "SizeOfOptionalHeader": pe.FILE_HEADER.SizeOfOptionalHeader,
        "Characteristics": pe.FILE_HEADER.Characteristics,

        "MajorLinkerVersion": pe.OPTIONAL_HEADER.MajorLinkerVersion,
        "MinorLinkerVersion": pe.OPTIONAL_HEADER.MinorLinkerVersion,
        "SizeOfCode": pe.OPTIONAL_HEADER.SizeOfCode,
        "SizeOfInitializedData": pe.OPTIONAL_HEADER.SizeOfInitializedData,
        "SizeOfUninitializedData": pe.OPTIONAL_HEADER.SizeOfUninitializedData,
        "AddressOfEntryPoint": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
        "BaseOfCode": pe.OPTIONAL_HEADER.BaseOfCode,
        "ImageBase": pe.OPTIONAL_HEADER.ImageBase,
        "SectionAlignment": pe.OPTIONAL_HEADER.SectionAlignment,
        "FileAlignment": pe.OPTIONAL_HEADER.FileAlignment,
        "MajorOperatingSystemVersion": pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
        "MinorOperatingSystemVersion": pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
        "MajorImageVersion": pe.OPTIONAL_HEADER.MajorImageVersion,
        "MinorImageVersion": pe.OPTIONAL_HEADER.MinorImageVersion,
        "MajorSubsystemVersion": pe.OPTIONAL_HEADER.MajorSubsystemVersion,
        "MinorSubsystemVersion": pe.OPTIONAL_HEADER.MinorSubsystemVersion,
        "SizeOfImage": pe.OPTIONAL_HEADER.SizeOfImage,
        "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
        "CheckSum": pe.OPTIONAL_HEADER.CheckSum,
        "Subsystem": pe.OPTIONAL_HEADER.Subsystem,
        "DllCharacteristics": pe.OPTIONAL_HEADER.DllCharacteristics,
        "SizeOfStackReserve": pe.OPTIONAL_HEADER.SizeOfStackReserve,
        "SizeOfStackCommit": pe.OPTIONAL_HEADER.SizeOfStackCommit,
        "SizeOfHeapReserve": pe.OPTIONAL_HEADER.SizeOfHeapReserve,
        "SizeOfHeapCommit": pe.OPTIONAL_HEADER.SizeOfHeapCommit,
    }

    # Section-based derived features
    entropies = []
    virtual_sizes = []
    raw_sizes = []
    for section in pe.sections:
        data = section.get_data()
        entropies.append(calculate_entropy(data))
        virtual_sizes.append(section.Misc_VirtualSize)
        raw_sizes.append(section.SizeOfRawData)

    if entropies:
        features["SectionMinEntropy"] = min(entropies)
        features["SectionMaxEntropy"] = max(entropies)
        features["SectionMeanEntropy"] = sum(entropies) / len(entropies)
    else:
        features["SectionMinEntropy"] = 0
        features["SectionMaxEntropy"] = 0
        features["SectionMeanEntropy"] = 0

    if virtual_sizes:
        features["SectionMinVirtualSize"] = min(virtual_sizes)
        features["SectionMaxVirtualSize"] = max(virtual_sizes)
    else:
        features["SectionMinVirtualSize"] = 0
        features["SectionMaxVirtualSize"] = 0

    features["SectionCount"] = len(pe.sections)

    return pd.DataFrame([features])

def extract_features(file_path, selected_features_path='models/selected_features.pkl', scaler_path=None):
    raw_features = extract_raw_features(file_path)
    selected_features = joblib.load(selected_features_path)

    # Fill missing cols if any
    for f in selected_features:
        if f not in raw_features.columns:
            raw_features[f] = 0

    final_features = raw_features[selected_features]

    # Optional: apply saved scaler
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        final_features = pd.DataFrame(scaler.transform(final_features), columns=selected_features)

    return final_features
