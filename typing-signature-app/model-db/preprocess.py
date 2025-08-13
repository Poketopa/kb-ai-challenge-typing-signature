import subprocess
import os
from pathlib import Path
import sys
sys.path.append(str((Path(__file__)/"utils").resolve()))

try:
    from utils.config import Config
except ImportError:
    # config.py가 없는 경우 기본 설정 사용
    class Config:
        def get_config_dict(self):
            return {
                "data": {
                    "keystroke_sequence_len": 50,
                    "aalto": {
                        "dataset_url": "https://userinterfaces.aalto.fi/typing37k/data/csv_raw_and_processed.zip"
                    }
                }
            }
import pandas as pd
import pickle
import numpy as np

def download_dataset():
    """Aalto DB 데이터셋 다운로드 (이미 다운로드된 경우 건너뛰기)"""
    data_raw_path = r"C:\Users\융합인재센터21\Desktop\csv_raw_and_processed\Data_Raw"
    keystrokes_path = f"{data_raw_path}\\keystrokes.csv"
    test_sections_path = f"{data_raw_path}\\test_sections.csv"
    
    print(f"Checking dataset path: {data_raw_path}")
    print(f"Looking for keystrokes.csv at: {keystrokes_path}")
    print(f"Looking for test_sections.csv at: {test_sections_path}")
    
    if os.path.exists(keystrokes_path) and os.path.exists(test_sections_path):
        print("Dataset already exists. Skipping download...")
        return data_raw_path
    else:
        print("Dataset files not found. Please check the path.")
        print(f"Keystrokes file exists: {os.path.exists(keystrokes_path)}")
        print(f"Test sections file exists: {os.path.exists(test_sections_path)}")
        return None

def extract_features(data):
    """키스트로크 데이터에서 3개 특징 추출"""
    print("Extracting features from keystroke data...")
    
    # 사용자별 세션 그룹화
    grouped = data.groupby("user_id")
    data_dict = {x: group for x, group in grouped}
    
    # 각 사용자별로 세션 그룹화
    for user in data_dict:
        data_dict[user] = [group[["press_time", "release_time", "key_code"]].to_numpy() 
                           for x, group in data_dict[user].groupby("session_id")]
    
    # 15개 세션이 없는 사용자 제거
    removing_users = []
    for user in data_dict:
        if len(data_dict[user]) != 15:
            removing_users.append(user)
    
    for key in removing_users:
        data_dict.pop(key, None)
    
    print(f"Total users after filtering: {len(data_dict)}")
    
    # 특징 추출 및 정규화
    processed_data = []
    for user_id, sessions in data_dict.items():
        user_sessions = []
        for session in sessions:
            session_features = []
            
            for i in range(len(session)):
                # 기본 특징
                press_time = session[i][0]
                release_time = session[i][1]
                key_code = session[i][2]
                
                # 1. Dwell Time (키 누름 시간)
                dwell_time = release_time - press_time
                
                # 2. Flight Time (키 놓음 시간)
                flight_time = 0.0
                if i < len(session) - 1:
                    flight_time = session[i+1][0] - release_time
                
                # 3. Press-Press Interval (연속 키 간격)
                press_interval = 0.0
                if i < len(session) - 1:
                    press_interval = session[i+1][0] - press_time
                
                # 정규화 (밀리초를 초 단위로)
                dwell_time = dwell_time / 1000.0
                flight_time = flight_time / 1000.0
                press_interval = press_interval / 1000.0
                
                # 특징 벡터 생성 [dwell_time, flight_time, press_interval]
                features = np.array([dwell_time, flight_time, press_interval])
                session_features.append(features)
            
            user_sessions.append(np.array(session_features))
        
        processed_data.append(user_sessions)
    
    return processed_data

def split_data(processed_data):
    """데이터를 훈련/검증/테스트로 분할"""
    print("Splitting data into train/validation/test sets...")
    
    total_users = len(processed_data)
    train_end = total_users - 1050
    val_end = total_users - 1000
    
    training_data = processed_data[:train_end]
    validation_data = processed_data[train_end:val_end]
    testing_data = processed_data[val_end:]
    
    print(f"Training users: {len(training_data)}")
    print(f"Validation users: {len(validation_data)}")
    print(f"Testing users: {len(testing_data)}")
    
    return training_data, validation_data, testing_data

def save_data(training_data, validation_data, testing_data):
    """전처리된 데이터를 pickle 파일로 저장"""
    print("Saving preprocessed data...")
    
    # 훈련 데이터 저장
    with open("data/training_data.pickle", 'wb') as f:
        pickle.dump(training_data, f)
    
    # 검증 데이터 저장
    with open("data/validation_data.pickle", 'wb') as f:
        pickle.dump(validation_data, f)
    
    # 테스트 데이터 저장
    with open("data/testing_data.pickle", 'wb') as f:
        pickle.dump(testing_data, f)
    
    print("Data saved successfully!")

def main():
    """메인 전처리 파이프라인"""
    print("Starting keystroke data preprocessing...")
    
    # 데이터셋 경로 확인
    data_raw_path = download_dataset()
    if data_raw_path is None:
        print("Error: Dataset not found!")
        return
    
    # 데이터 로딩
    print("Loading keystroke data...")
    keystrokes_path = f"{data_raw_path}\\keystrokes.csv"
    test_sections_path = f"{data_raw_path}\\test_sections.csv"
    
    # 헤더 파일 먼저 확인
    keystrokes_header_path = f"{data_raw_path}\\keystrokes_header.csv"
    test_sections_header_path = f"{data_raw_path}\\test_sections_header.csv"
    
    # 헤더 정보 로딩
    if os.path.exists(keystrokes_header_path):
        keystroke_headers = pd.read_csv(keystrokes_header_path).Field.tolist()
    else:
        # 기본 헤더 사용
        keystroke_headers = ["TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"]
    
    if os.path.exists(test_sections_header_path):
        test_section_headers = pd.read_csv(test_sections_header_path).Field.tolist()
    else:
        # 기본 헤더 사용
        test_section_headers = ["TEST_SECTION_ID", "PARTICIPANT_ID"]
    
    # 데이터 로딩
    print(f"Loading keystrokes from: {keystrokes_path}")
    data = pd.read_csv(keystrokes_path, names=keystroke_headers, encoding='latin-1', on_bad_lines='skip')
    
    print(f"Loading test sections from: {test_sections_path}")
    test_sections = pd.read_csv(test_sections_path, names=test_section_headers, encoding='latin-1', on_bad_lines='skip')
    
    # 필요한 컬럼만 선택
    data = data[["TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"]]
    test_sections = test_sections[["TEST_SECTION_ID", "PARTICIPANT_ID"]]
    
    print(f"Keystrokes data shape: {data.shape}")
    print(f"Test sections data shape: {test_sections.shape}")
    
    # 데이터 병합
    print("Merging data...")
    data = data.merge(test_sections, on="TEST_SECTION_ID")
    print(f"Merged data shape: {data.shape}")
    
    data.rename(columns={
        'TEST_SECTION_ID': 'session_id',
        'PARTICIPANT_ID': 'user_id',
        'PRESS_TIME': 'press_time',
        'RELEASE_TIME': 'release_time',
        'KEYCODE': 'key_code'
    }, inplace=True)
    
    # NaN 값 검증
    print("Checking for NaN values...")
    nan_count = data.isnull().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values. Removing rows with NaN...")
        data = data.dropna()
    
    print(f"Final data shape: {data.shape}")
    
    # data 폴더 생성
    os.makedirs("data", exist_ok=True)
    
    # 특징 추출
    processed_data = extract_features(data)
    
    # 데이터 분할
    training_data, validation_data, testing_data = split_data(processed_data)
    
    # 데이터 저장
    save_data(training_data, validation_data, testing_data)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main() 