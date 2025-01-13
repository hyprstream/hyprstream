use std::{collections, fmt::Debug};

use rusqlite::types::{ToSqlOutput, Value, ValueRef};
use serde_derive::{Deserialize, Serialize};

use crate::Error;

use super::to_params_named_with_fields;

fn make_connection() -> rusqlite::Connection {
	make_connection_with_spec(
		"
		f_integer INT CHECK(typeof(f_integer) IN ('integer', 'null')),
		f_real REAL CHECK(typeof(f_real) IN ('real', 'null')),
		f_text TEXT CHECK(typeof(f_text) IN ('text', 'null')),
		f_blob BLOB CHECK(typeof(f_blob) IN ('blob', 'null')),
		f_null INT CHECK(typeof(f_integer) IN ('integer', 'null'))
	",
	)
}

fn make_connection_with_spec(table_spec: &str) -> rusqlite::Connection {
	let con = rusqlite::Connection::open_in_memory().unwrap();
	con.execute(&format!("CREATE TABLE test({})", table_spec), []).unwrap();
	con
}

fn test_value_same<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + Debug + Clone>(db_type: &str, src: &T) {
	test_values(db_type, &src.clone(), src)
}

fn test_values<D: serde::de::DeserializeOwned + PartialEq + Debug>(
	db_type: &str,
	value_ser: &impl serde::Serialize,
	value_de: &D,
) {
	test_values_with_cmp_fn::<_, _, &dyn Fn(&D, &D) -> bool>(db_type, value_ser, value_de, None)
}

fn test_ser_err<S: serde::Serialize, F: Fn(&Error) -> bool>(value: &S, err_check_fn: F) {
	match super::to_params(value) {
		Err(e) => assert!(err_check_fn(&e), "Error raised was not of the correct type, got: {}", e),
		_ => panic!("Error was not raised"),
	}
}

fn test_values_with_cmp_fn<S, D, F>(db_type: &str, value_ser: &S, value_de: &D, comparison_fn: Option<F>)
where
	S: serde::Serialize,
	D: serde::de::DeserializeOwned + PartialEq + Debug,
	F: Fn(&D, &D) -> bool,
{
	let con = make_connection_with_spec(&format!("test_column {}", db_type));
	// serialization
	con.execute(
		"INSERT INTO test(test_column) VALUES(?)",
		super::to_params(value_ser).unwrap(),
	)
	.unwrap();
	// deserialization
	let mut stmt = con.prepare("SELECT * FROM test").unwrap();
	let res = stmt.query_and_then([], super::from_row::<D>).unwrap();
	for row in res {
		let row = row.unwrap();
		match comparison_fn {
			None => assert_eq!(row, *value_de),
			Some(ref comparison_fn) => assert!(
				comparison_fn(&row, value_de),
				"value after deserialization is not the same as before"
			),
		}
	}
}

#[test]
fn test_bool() {
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &false);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &true);
}

#[test]
fn test_int() {
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &0_i8);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &-9881_i16);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &16526_i32);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &-18968298731236_i64);
}

#[test]
fn test_uint() {
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &112_u8);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &7162u16);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &98172983_u32);
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &98169812698712987_u64);
	test_ser_err(&u64::MAX, |err| matches!(*err, super::Error::ValueTooLarge(..)));
}

#[test]
fn test_float() {
	test_value_same("REAL CHECK(typeof(test_column) == 'real')", &0.3_f32);
	test_value_same("REAL CHECK(typeof(test_column) == 'real')", &-54.7612_f64);
	test_value_same("REAL CHECK(typeof(test_column) == 'real')", &f64::NEG_INFINITY);
	test_value_same("REAL CHECK(typeof(test_column) == 'real')", &f64::INFINITY);
	test_value_same("REAL CHECK(typeof(test_column) == 'real')", &f32::NEG_INFINITY);
	test_value_same("REAL CHECK(typeof(test_column) == 'real')", &f32::INFINITY);
	// can't compare 2 NaN's directly, so using custom comparison function
	test_values_with_cmp_fn(
		"REAL CHECK(typeof(test_column) == 'null')",
		&f64::NAN,
		&f64::NAN,
		Some(|db: &f64, value: &f64| db.is_nan() && value.is_nan()),
	);
	test_values_with_cmp_fn(
		"REAL CHECK(typeof(test_column) == 'null')",
		&f32::NAN,
		&f32::NAN,
		Some(|db: &f32, value: &f32| db.is_nan() && value.is_nan()),
	);
}

#[test]
fn test_string() {
	test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &'a');
	test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &"test string".to_owned());
	test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &"Ünicódé".to_owned());
	let val = "test string";
	test_values("TEXT CHECK(typeof(test_column) == 'text')", &val, &val.to_string());
}

#[test]
fn test_bytes() {
	let val = b"123456";
	test_values(
		"BLOB CHECK(typeof(test_column) == 'blob')",
		&serde_bytes::Bytes::new(val),
		&val.to_vec(),
	);
	test_values(
		"BLOB CHECK(typeof(test_column) == 'blob')",
		&serde_bytes::Bytes::new(val),
		&serde_bytes::ByteBuf::from(val.to_vec()),
	);
}

#[test]
fn test_nullable() {
	test_value_same("INT CHECK(typeof(test_column) == 'integer')", &Some(18));
	test_value_same::<Option<i64>>("INT CHECK(typeof(test_column) == 'null')", &None);
	test_value_same("INT CHECK(typeof(test_column) == 'null')", &());
}

#[test]
fn test_enum() {
	{
		#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
		enum Test {
			A,
			B,
			C,
		}
		test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &Test::A);
		test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &Test::B);
		test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &Test::C);
	}
}

#[test]
fn test_map() {
	{
		let con = make_connection_with_spec(
			"
			field_1 INT CHECK(typeof(field_1) == 'integer'),
			field_2 INT CHECK(typeof(field_2) == 'integer'),
			field_3 INT CHECK(typeof(field_3) == 'integer')
		",
		);
		// serialization
		let mut src = collections::HashMap::<String, i64>::new();
		src.insert("field_2".into(), 2);
		src.insert("field_1".into(), 1);
		src.insert("field_3".into(), 3);
		con.execute(
			"INSERT INTO test VALUES(:field_1, :field_2, :field_3)",
			super::to_params_named(&src).unwrap().to_slice().as_slice(),
		)
		.unwrap();
		// deserialization with columns
		let mut stmt = con.prepare("SELECT * FROM test").unwrap();
		{
			let columns = super::columns_from_statement(&stmt);
			let mut res = stmt
				.query_and_then([], |row| {
					super::from_row_with_columns::<collections::HashMap<String, i64>>(row, &columns)
				})
				.unwrap();
			assert_eq!(res.next().unwrap().unwrap(), src);
		}
		// deserialization without columns
		{
			let mut res = stmt
				.query_and_then([], super::from_row::<collections::HashMap<String, i64>>)
				.unwrap();
			assert_eq!(res.next().unwrap().unwrap(), src);
		}
	}

	{
		let con = make_connection_with_spec(
			"
			a INT CHECK(typeof(a) == 'integer'),
			b INT CHECK(typeof(b) == 'integer'),
			c INT CHECK(typeof(c) == 'integer')
		",
		);
		// serialization
		let mut src = collections::HashMap::<char, i64>::new();
		src.insert('a', 2);
		src.insert('b', 1);
		src.insert('c', 3);
		con.execute(
			"INSERT INTO test VALUES(:a, :b, :c)",
			super::to_params_named(&src).unwrap().to_slice().as_slice(),
		)
		.unwrap();
		let mut stmt = con.prepare("SELECT * FROM test").unwrap();
		// deserialization with columns
		{
			let columns = super::columns_from_statement(&stmt);
			let mut res = stmt
				.query_and_then([], |row| {
					super::from_row_with_columns::<collections::HashMap<char, i64>>(row, &columns)
				})
				.unwrap();
			assert_eq!(res.next().unwrap().unwrap(), src);
		}
		// deserialization with custom columns
		{
			let columns = vec!["a".to_owned(), "b".to_owned()];
			let mut res = stmt
				.query_and_then([], |row| {
					super::from_row_with_columns::<collections::HashMap<char, i64>>(row, &columns)
				})
				.unwrap();
			let mut src = src.clone();
			src.remove(&'c');
			assert_eq!(res.next().unwrap().unwrap(), src);
		}
	}
}

#[test]
fn test_tuple() {
	let con = make_connection();
	type Test = (i64, f64, String, Vec<u8>, Option<i64>);
	// serialization
	let src: Test = (34, 76.4, "the test".into(), vec![10, 20, 30], Some(9));
	con.execute("INSERT INTO test VALUES(?, ?, ?, ?, ?)", super::to_params(&src).unwrap())
		.unwrap();
	let mut stmt = con.prepare("SELECT * FROM test").unwrap();
	// deserialization with columns
	{
		let columns = super::columns_from_statement(&stmt);
		let mut res = stmt
			.query_and_then([], |row| super::from_row_with_columns::<Test>(row, &columns))
			.unwrap();
		assert_eq!(res.next().unwrap().unwrap(), src);
	}

	// deserialization without columns
	{
		let mut res = stmt.query_and_then([], super::from_row::<Test>).unwrap();
		assert_eq!(res.next().unwrap().unwrap(), src);
	}
}

#[test]
fn test_struct() {
	{
		#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
		struct Test(i64);
		test_value_same("INT CHECK(typeof(test_column) == 'integer')", &Test(891287912));
	}
	{
		#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
		struct Test;
		test_value_same("TEXT CHECK(typeof(test_column) == 'text')", &Test);
	}
	{
		let con = make_connection();
		#[derive(Deserialize, Serialize, Debug, PartialEq)]
		struct Test {
			f_integer: i64,
			f_real: f64,
			f_text: String,
			f_blob: Vec<u8>,
			f_null: Option<i64>,
		}
		#[derive(Serialize)]
		struct TestRef<'a> {
			f_integer: i64,
			f_real: f64,
			f_text: &'a str,
			f_blob: &'a [u8],
			f_null: Option<i64>,
		}
		// serialization
		let src = Test {
			f_integer: 10,
			f_real: 65.3,
			f_text: "the test".into(),
			f_blob: vec![0, 1, 2],
			f_null: None,
		};
		let src_ref = TestRef {
			f_integer: src.f_integer,
			f_real: src.f_real,
			f_text: &src.f_text,
			f_blob: &src.f_blob,
			f_null: src.f_null,
		};
		con.execute(
			"INSERT INTO test VALUES(:f_integer, :f_real, :f_text, :f_blob, :f_null)",
			super::to_params_named(src_ref).unwrap().to_slice().as_slice(),
		)
		.unwrap();
		// deserialization with columns
		let mut stmt = con.prepare("SELECT * FROM test").unwrap();
		{
			let columns = super::columns_from_statement(&stmt);
			let mut res = stmt
				.query_and_then([], |row| super::from_row_with_columns::<Test>(row, &columns))
				.unwrap();
			assert_eq!(res.next().unwrap().unwrap(), src);
		}
		// deserialization without columns
		{
			let mut res = stmt.query_and_then([], super::from_row::<Test>).unwrap();
			assert_eq!(res.next().unwrap().unwrap(), src);
		}
	}

	{
		let con = make_connection();
		#[derive(Deserialize, Serialize, Debug, PartialEq)]
		struct Test {
			#[serde(with = "serde_bytes")]
			f_blob: Vec<u8>,
			f_integer: i64,
			f_text: String,
			f_null: Option<i64>,
			f_real: f64,
		}
		// serialization
		let src = Test {
			f_blob: vec![5, 10, 15],
			f_integer: 10,
			f_real: -65.3,
			f_text: "".into(),
			f_null: Some(43),
		};
		con.execute(
			"INSERT INTO test VALUES(:f_integer, :f_real, :f_text, :f_blob, :f_null)",
			super::to_params_named(&src).unwrap().to_slice().as_slice(),
		)
		.unwrap();
		con.execute(
			"INSERT INTO test VALUES(:f_integer, :f_real, :f_text, :f_blob, :f_null)",
			super::to_params_named(&src).unwrap().to_slice().as_slice(),
		)
		.unwrap();
		// deserialization with columns
		let mut stmt = con.prepare("SELECT * FROM test").unwrap();
		let mut rows = stmt.query([]).unwrap();
		// deserialization
		let mut res = super::from_rows_ref::<Test>(&mut rows);
		assert_eq!(res.next().unwrap().unwrap(), src);
		assert_eq!(super::from_row::<Test>(rows.next().unwrap().unwrap()).unwrap(), src);
	}

	{
		let con = make_connection();
		#[derive(Deserialize, Serialize, Debug, PartialEq)]
		struct Test {
			#[serde(with = "serde_bytes")]
			f_blob: Vec<u8>,
			f_integer: i64,
			f_text: String,
			f_null: Option<i64>,
			f_real: f64,
		}
		// serialization
		let src = Test {
			f_blob: vec![5, 10, 15],
			f_integer: 10,
			f_real: -65.3,
			f_text: "".into(),
			f_null: Some(43),
		};
		con.execute(
			"INSERT INTO test VALUES(:f_integer, :f_real, :f_text, :f_blob, :f_null)",
			super::to_params_named(&src).unwrap().to_slice().as_slice(),
		)
		.unwrap();
		// deserialization
		let mut stmt = con.prepare("SELECT * FROM test").unwrap();
		let mut res = super::from_rows::<Test>(stmt.query([]).unwrap());
		assert_eq!(res.next().unwrap().unwrap(), src);
	}
}

#[test]
fn test_attrs() {
	let con = make_connection();
	#[derive(Serialize, Debug, PartialEq)]
	struct Test {
		#[serde(rename = "f_real")]
		custom_real: f64,
		f_integer: i64,
		f_blob: Vec<u8>,
		f_null: Option<i64>,
		f_text: String,
	}
	let src = Test {
		f_blob: vec![5, 10, 15],
		f_integer: 10,
		custom_real: -65.3,
		f_text: "test".into(),
		f_null: Some(43),
	};
	con.execute(
		"INSERT INTO test VALUES(:f_integer, :f_real, :f_text, :f_blob, :f_null)",
		super::to_params_named(&src).unwrap().to_slice().as_slice(),
	)
	.unwrap();
	#[derive(Deserialize, Debug, PartialEq)]
	struct TestDeser {
		f_blob: Vec<u8>,
		#[serde(alias = "f_real")]
		custom_real: f64,
		#[serde(default = "default_string")]
		f_text: String,
		#[serde(default)]
		f_integer: i64,
		f_null: Option<i64>,
	}

	fn default_string() -> String {
		"default".into()
	}

	let mut stmt = con.prepare("SELECT f_real, f_blob, f_null FROM test").unwrap();
	{
		let mut res = super::from_rows::<TestDeser>(stmt.query([]).unwrap());
		let row = res.next().unwrap().unwrap();
		assert_eq!(row.f_integer, i64::default());
		assert_eq!(row.custom_real, src.custom_real);
		assert_eq!(row.f_text, default_string());
		assert_eq!(row.f_blob, src.f_blob);
		assert_eq!(row.f_null, src.f_null);
	}
}

#[test]
fn test_deser_err() {
	let con = make_connection();
	#[derive(Serialize, Debug, PartialEq)]
	struct Ser {
		f_real: f64,
		f_text: String,
	}
	let src = Ser {
		f_real: -65.3,
		f_text: "test".to_string(),
	};
	con.execute(
		"INSERT INTO test(f_real, f_text) VALUES(:f_real, :f_text)",
		super::to_params_named(src).unwrap().to_slice().as_slice(),
	)
	.unwrap();
	#[derive(Deserialize, Debug, PartialEq)]
	struct Deser {
		f_real: f64,
		f_text: i64,
	}

	let mut stmt = con.prepare("SELECT f_real, f_text FROM test").unwrap();
	{
		let mut res = super::from_rows::<Deser>(stmt.query([]).unwrap());
		let err = res.next().unwrap();
		match err {
			Err(Error::Deserialization { column: Some(field), .. }) => {
				assert_eq!(field, "f_text")
			}
			_ => panic!("Unexpected result: {:?}", err),
		}
	}
}

#[test]
fn pluck_named() {
	#[derive(Debug, Serialize, Deserialize)]
	struct Food {
		id: i32,
		name: String,
		flavor: String,
		color: String,
		expiration: String,
	}

	let item = Food {
		id: 15,
		name: "Snickers bar".to_string(),
		flavor: "choco and caramel".to_string(),
		color: "brown".to_string(),
		expiration: "2021-04-05".to_string(),
	};

	let plucked = to_params_named_with_fields(item, &["flavor", "id"]).unwrap();
	let sqlified = plucked
		.iter()
		.map(|(n, x)| (n.as_str(), x.to_sql().unwrap()))
		.collect::<Vec<_>>();

	assert_eq!(
		vec![
			(":id", ToSqlOutput::Owned(Value::Integer(15))),
			(
				":flavor",
				ToSqlOutput::Borrowed(ValueRef::Text("choco and caramel".as_bytes()))
			),
		],
		sqlified
	);
}
