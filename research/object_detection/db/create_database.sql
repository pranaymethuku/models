/* 
  Schema for creating database
  Tables - 
    Detection - (id^, file_path, labeled_file_path, confidence, label**, tier** , model, time_stamp)
    Category - (label*, tier*)

    * - Primary key
    ** - Foreign key
    ^ - Surrogate (implicit) key
 */
PRAGMA foreign_keys = ON;

CREATE TABLE Detection (
  file_path varchar(100) NOT NULL,
  labeled_file_path varchar(100) NOT NULL,
  confidence real NOT NULL,
  label varchar(50) NOT NULL,
  tier int NOT NULL,
  model varchar(50) NOT NULL,
  time_stamp text,
  FOREIGN KEY (label, tier) REFERENCES Category(label, tier)
);

CREATE TABLE Category (
  label varchar(50) NOT NULL,
  tier int NOT NULL,
  PRIMARY KEY (label, tier)
);
