/* 
  Schema for creating database
  Tables - 
    Vehicle - (id^, img_path, labeled_img_path, confidence, label**, tier** , loc, time_stamp)
    Category - (label*, tier*)

    * - Primary key
    ** - Foreign key
    ^ - Surrogate (implicit) key
 */
PRAGMA foreign_keys = ON;

CREATE TABLE Vehicle (
  img_path varchar(100) NOT NULL,
  labeled_img_path varchar(100) NOT NULL,
  confidence real NOT NULL,
  label varchar(50) NOT NULL,
  tier real NOT NULL,
  loc varchar(50),
  time_stamp text,
  FOREIGN KEY (label, tier) REFERENCES Category(label, tier)
);

CREATE TABLE Category (
  label varchar(50) NOT NULL,
  tier real NOT NULL,
  PRIMARY KEY (label, tier)
);
