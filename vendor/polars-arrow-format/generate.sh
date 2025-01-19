for NAME in Tensor SparseTensor Schema Message File
do
	curl https://raw.githubusercontent.com/apache/arrow/master/format/$NAME.fbs -o $NAME.fbs
done

planus rust *.fbs -o src/ipc/generated.rs
