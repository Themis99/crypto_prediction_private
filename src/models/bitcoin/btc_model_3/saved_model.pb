ë5
Ø
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
 
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
©
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ä/
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
¨
$tcn/residual_block_0/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$tcn/residual_block_0/conv1D_0/kernel
¡
8tcn/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_0/kernel*"
_output_shapes
:@*
dtype0

"tcn/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_0/conv1D_0/bias

6tcn/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_0/conv1D_0/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_0/conv1D_1/kernel
¡
8tcn/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_0/conv1D_1/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_0/conv1D_1/bias

6tcn/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_0/conv1D_1/bias*
_output_shapes
:@*
dtype0
¶
+tcn/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+tcn/residual_block_0/matching_conv1D/kernel
¯
?tcn/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp+tcn/residual_block_0/matching_conv1D/kernel*"
_output_shapes
:@*
dtype0
ª
)tcn/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)tcn/residual_block_0/matching_conv1D/bias
£
=tcn/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp)tcn/residual_block_0/matching_conv1D/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_1/conv1D_0/kernel
¡
8tcn/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_0/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_1/conv1D_0/bias

6tcn/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_1/conv1D_0/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_1/conv1D_1/kernel
¡
8tcn/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_1/conv1D_1/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_1/conv1D_1/bias

6tcn/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_1/conv1D_1/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_2/conv1D_0/kernel
¡
8tcn/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_0/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_2/conv1D_0/bias

6tcn/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_2/conv1D_0/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_2/conv1D_1/kernel
¡
8tcn/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_2/conv1D_1/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_2/conv1D_1/bias

6tcn/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_2/conv1D_1/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_3/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_3/conv1D_0/kernel
¡
8tcn/residual_block_3/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_0/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_3/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_3/conv1D_0/bias

6tcn/residual_block_3/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_3/conv1D_0/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_3/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_3/conv1D_1/kernel
¡
8tcn/residual_block_3/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_3/conv1D_1/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_3/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_3/conv1D_1/bias

6tcn/residual_block_3/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_3/conv1D_1/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_4/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_4/conv1D_0/kernel
¡
8tcn/residual_block_4/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_4/conv1D_0/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_4/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_4/conv1D_0/bias

6tcn/residual_block_4/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_4/conv1D_0/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_4/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_4/conv1D_1/kernel
¡
8tcn/residual_block_4/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_4/conv1D_1/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_4/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_4/conv1D_1/bias

6tcn/residual_block_4/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_4/conv1D_1/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_5/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_5/conv1D_0/kernel
¡
8tcn/residual_block_5/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_5/conv1D_0/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_5/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_5/conv1D_0/bias

6tcn/residual_block_5/conv1D_0/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_5/conv1D_0/bias*
_output_shapes
:@*
dtype0
¨
$tcn/residual_block_5/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$tcn/residual_block_5/conv1D_1/kernel
¡
8tcn/residual_block_5/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp$tcn/residual_block_5/conv1D_1/kernel*"
_output_shapes
:@@*
dtype0

"tcn/residual_block_5/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"tcn/residual_block_5/conv1D_1/bias

6tcn/residual_block_5/conv1D_1/bias/Read/ReadVariableOpReadVariableOp"tcn/residual_block_5/conv1D_1/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
¶
+Adam/tcn/residual_block_0/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/tcn/residual_block_0/conv1D_0/kernel/m
¯
?Adam/tcn/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_0/kernel/m*"
_output_shapes
:@*
dtype0
ª
)Adam/tcn/residual_block_0/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_0/conv1D_0/bias/m
£
=Adam/tcn/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_0/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_0/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_0/conv1D_1/kernel/m
¯
?Adam/tcn/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_1/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_0/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_0/conv1D_1/bias/m
£
=Adam/tcn/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_1/bias/m*
_output_shapes
:@*
dtype0
Ä
2Adam/tcn/residual_block_0/matching_conv1D/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42Adam/tcn/residual_block_0/matching_conv1D/kernel/m
½
FAdam/tcn/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/tcn/residual_block_0/matching_conv1D/kernel/m*"
_output_shapes
:@*
dtype0
¸
0Adam/tcn/residual_block_0/matching_conv1D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/tcn/residual_block_0/matching_conv1D/bias/m
±
DAdam/tcn/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOpReadVariableOp0Adam/tcn/residual_block_0/matching_conv1D/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_1/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_1/conv1D_0/kernel/m
¯
?Adam/tcn/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_0/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_1/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_1/conv1D_0/bias/m
£
=Adam/tcn/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_0/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_1/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_1/conv1D_1/kernel/m
¯
?Adam/tcn/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_1/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_1/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_1/conv1D_1/bias/m
£
=Adam/tcn/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_1/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_2/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_2/conv1D_0/kernel/m
¯
?Adam/tcn/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_0/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_2/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_2/conv1D_0/bias/m
£
=Adam/tcn/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_0/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_2/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_2/conv1D_1/kernel/m
¯
?Adam/tcn/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_1/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_2/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_2/conv1D_1/bias/m
£
=Adam/tcn/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_1/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_3/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_3/conv1D_0/kernel/m
¯
?Adam/tcn/residual_block_3/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_0/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_3/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_3/conv1D_0/bias/m
£
=Adam/tcn/residual_block_3/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_0/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_3/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_3/conv1D_1/kernel/m
¯
?Adam/tcn/residual_block_3/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_1/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_3/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_3/conv1D_1/bias/m
£
=Adam/tcn/residual_block_3/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_1/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_4/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_4/conv1D_0/kernel/m
¯
?Adam/tcn/residual_block_4/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_4/conv1D_0/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_4/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_4/conv1D_0/bias/m
£
=Adam/tcn/residual_block_4/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_4/conv1D_0/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_4/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_4/conv1D_1/kernel/m
¯
?Adam/tcn/residual_block_4/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_4/conv1D_1/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_4/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_4/conv1D_1/bias/m
£
=Adam/tcn/residual_block_4/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_4/conv1D_1/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_5/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_5/conv1D_0/kernel/m
¯
?Adam/tcn/residual_block_5/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_5/conv1D_0/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_5/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_5/conv1D_0/bias/m
£
=Adam/tcn/residual_block_5/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_5/conv1D_0/bias/m*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_5/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_5/conv1D_1/kernel/m
¯
?Adam/tcn/residual_block_5/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_5/conv1D_1/kernel/m*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_5/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_5/conv1D_1/bias/m
£
=Adam/tcn/residual_block_5/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_5/conv1D_1/bias/m*
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
¶
+Adam/tcn/residual_block_0/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/tcn/residual_block_0/conv1D_0/kernel/v
¯
?Adam/tcn/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_0/kernel/v*"
_output_shapes
:@*
dtype0
ª
)Adam/tcn/residual_block_0/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_0/conv1D_0/bias/v
£
=Adam/tcn/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_0/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_0/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_0/conv1D_1/kernel/v
¯
?Adam/tcn/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_0/conv1D_1/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_0/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_0/conv1D_1/bias/v
£
=Adam/tcn/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_0/conv1D_1/bias/v*
_output_shapes
:@*
dtype0
Ä
2Adam/tcn/residual_block_0/matching_conv1D/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42Adam/tcn/residual_block_0/matching_conv1D/kernel/v
½
FAdam/tcn/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/tcn/residual_block_0/matching_conv1D/kernel/v*"
_output_shapes
:@*
dtype0
¸
0Adam/tcn/residual_block_0/matching_conv1D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/tcn/residual_block_0/matching_conv1D/bias/v
±
DAdam/tcn/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOpReadVariableOp0Adam/tcn/residual_block_0/matching_conv1D/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_1/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_1/conv1D_0/kernel/v
¯
?Adam/tcn/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_0/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_1/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_1/conv1D_0/bias/v
£
=Adam/tcn/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_0/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_1/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_1/conv1D_1/kernel/v
¯
?Adam/tcn/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_1/conv1D_1/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_1/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_1/conv1D_1/bias/v
£
=Adam/tcn/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_1/conv1D_1/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_2/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_2/conv1D_0/kernel/v
¯
?Adam/tcn/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_0/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_2/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_2/conv1D_0/bias/v
£
=Adam/tcn/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_0/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_2/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_2/conv1D_1/kernel/v
¯
?Adam/tcn/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_2/conv1D_1/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_2/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_2/conv1D_1/bias/v
£
=Adam/tcn/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_2/conv1D_1/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_3/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_3/conv1D_0/kernel/v
¯
?Adam/tcn/residual_block_3/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_0/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_3/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_3/conv1D_0/bias/v
£
=Adam/tcn/residual_block_3/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_0/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_3/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_3/conv1D_1/kernel/v
¯
?Adam/tcn/residual_block_3/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_3/conv1D_1/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_3/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_3/conv1D_1/bias/v
£
=Adam/tcn/residual_block_3/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_3/conv1D_1/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_4/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_4/conv1D_0/kernel/v
¯
?Adam/tcn/residual_block_4/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_4/conv1D_0/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_4/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_4/conv1D_0/bias/v
£
=Adam/tcn/residual_block_4/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_4/conv1D_0/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_4/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_4/conv1D_1/kernel/v
¯
?Adam/tcn/residual_block_4/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_4/conv1D_1/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_4/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_4/conv1D_1/bias/v
£
=Adam/tcn/residual_block_4/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_4/conv1D_1/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_5/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_5/conv1D_0/kernel/v
¯
?Adam/tcn/residual_block_5/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_5/conv1D_0/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_5/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_5/conv1D_0/bias/v
£
=Adam/tcn/residual_block_5/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_5/conv1D_0/bias/v*
_output_shapes
:@*
dtype0
¶
+Adam/tcn/residual_block_5/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+Adam/tcn/residual_block_5/conv1D_1/kernel/v
¯
?Adam/tcn/residual_block_5/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tcn/residual_block_5/conv1D_1/kernel/v*"
_output_shapes
:@@*
dtype0
ª
)Adam/tcn/residual_block_5/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/tcn/residual_block_5/conv1D_1/bias/v
£
=Adam/tcn/residual_block_5/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tcn/residual_block_5/conv1D_1/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
ëü
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¥ü
valueüBü Bü

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
å
skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
residual_block_3
residual_block_4
residual_block_5
slicer_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
ô
$iter

%beta_1

&beta_2
	'decay
(learning_ratemÖm×)mØ*mÙ+mÚ,mÛ-mÜ.mÝ/mÞ0mß1mà2má3mâ4mã5mä6må7mæ8mç9mè:mé;mê<më=mì>mí?mî@mïAmðBmñvòvó)vô*võ+vö,v÷-vø.vù/vú0vû1vü2vý3vþ4vÿ5v6v7v8v9v:v;v<v=v>v?v@vAvBv*
Ú
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
26
27*
Ú
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
26
27*
* 
°
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

Hserving_default* 
* 
.
0
1
2
3
4
5*
* 
å

Ilayers
Jshape_match_conv
Kfinal_activation
Lconv1D_0
MAct_Conv1D_0
N
SDropout_0
Oconv1D_1
PAct_Conv1D_1
Q
SDropout_1
RAct_Conv_Blocks
Jmatching_conv1D
KAct_Res_Block
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*
ç

Ylayers
Zshape_match_conv
[final_activation
\conv1D_0
]Act_Conv1D_0
^
SDropout_0
_conv1D_1
`Act_Conv1D_1
a
SDropout_1
bAct_Conv_Blocks
Zmatching_identity
[Act_Res_Block
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*
ç

ilayers
jshape_match_conv
kfinal_activation
lconv1D_0
mAct_Conv1D_0
n
SDropout_0
oconv1D_1
pAct_Conv1D_1
q
SDropout_1
rAct_Conv_Blocks
jmatching_identity
kAct_Res_Block
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses*
ð

ylayers
zshape_match_conv
{final_activation
|conv1D_0
}Act_Conv1D_0
~
SDropout_0
conv1D_1
Act_Conv1D_1

SDropout_1
Act_Conv_Blocks
zmatching_identity
{Act_Res_Block
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
ù
layers
shape_match_conv
final_activation
conv1D_0
Act_Conv1D_0

SDropout_0
conv1D_1
Act_Conv1D_1

SDropout_1
Act_Conv_Blocks
matching_identity
Act_Res_Block
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
ù
layers
shape_match_conv
final_activation
conv1D_0
Act_Conv1D_0

SDropout_0
conv1D_1
 Act_Conv1D_1
¡
SDropout_1
¢Act_Conv_Blocks
matching_identity
Act_Res_Block
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses*

©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses* 
Ê
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25*
Ê
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$tcn/residual_block_0/conv1D_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"tcn/residual_block_0/conv1D_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$tcn/residual_block_0/conv1D_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"tcn/residual_block_0/conv1D_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+tcn/residual_block_0/matching_conv1D/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)tcn/residual_block_0/matching_conv1D/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$tcn/residual_block_1/conv1D_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"tcn/residual_block_1/conv1D_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$tcn/residual_block_1/conv1D_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"tcn/residual_block_1/conv1D_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_2/conv1D_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_2/conv1D_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_2/conv1D_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_2/conv1D_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_3/conv1D_0/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_3/conv1D_0/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_3/conv1D_1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_3/conv1D_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_4/conv1D_0/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_4/conv1D_0/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_4/conv1D_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_4/conv1D_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_5/conv1D_0/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_5/conv1D_0/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tcn/residual_block_5/conv1D_1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"tcn/residual_block_5/conv1D_1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
,
¹0
º1
»2
¼3
½4*
* 
* 
* 
5
L0
M1
N2
O3
P4
Q5
R6*
¬

-kernel
.bias
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses*

Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses* 
¬

)kernel
*bias
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses*

Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses* 
¬
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú_random_generator
Û__call__
+Ü&call_and_return_all_conditional_losses* 
¬

+kernel
,bias
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses*

ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç__call__
+è&call_and_return_all_conditional_losses* 
¬
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í_random_generator
î__call__
+ï&call_and_return_all_conditional_losses* 

ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
ô__call__
+õ&call_and_return_all_conditional_losses* 
.
)0
*1
+2
,3
-4
.5*
.
)0
*1
+2
,3
-4
.5*
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
5
\0
]1
^2
_3
`4
a5
b6*

û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

/kernel
0bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 
¬

1kernel
2bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses* 
¬
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª_random_generator
«__call__
+¬&call_and_return_all_conditional_losses* 

­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses* 
 
/0
01
12
23*
 
/0
01
12
23*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 
5
l0
m1
n2
o3
p4
q5
r6*

¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses* 

¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses* 
¬

3kernel
4bias
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses*

Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses* 
¬
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô_random_generator
Õ__call__
+Ö&call_and_return_all_conditional_losses* 
¬

5kernel
6bias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses*

Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses* 
¬
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç_random_generator
è__call__
+é&call_and_return_all_conditional_losses* 

ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses* 
 
30
41
52
63*
 
30
41
52
63*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*
* 
* 
8
|0
}1
~2
3
4
5
6*

õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses* 

û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses* 
¬

7kernel
8bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 
¬

9kernel
:bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤_random_generator
¥__call__
+¦&call_and_return_all_conditional_losses* 

§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses* 
 
70
81
92
:3*
 
70
81
92
:3*
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
<
0
1
2
3
4
5
6*

²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses* 

¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses* 
¬

;kernel
<bias
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses*

Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses* 
¬
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î_random_generator
Ï__call__
+Ð&call_and_return_all_conditional_losses* 
¬

=kernel
>bias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses*

×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses* 
¬
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á_random_generator
â__call__
+ã&call_and_return_all_conditional_losses* 

ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses* 
 
;0
<1
=2
>3*
 
;0
<1
=2
>3*
* 

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
<
0
1
2
3
 4
¡5
¢6*

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses* 

õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses* 
¬

?kernel
@bias
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 
¬

Akernel
Bbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+ &call_and_return_all_conditional_losses* 

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses* 
 
?0
@1
A2
B3*
 
?0
@1
A2
B3*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses* 
* 
* 
* 
5
0
1
2
3
4
5
6*
* 
* 
* 
* 
* 
* 
* 
* 
<

±total

²count
³	variables
´	keras_api*
<

µtotal

¶count
·	variables
¸	keras_api*
M

¹total

ºcount
»
_fn_kwargs
¼	variables
½	keras_api*
M

¾total

¿count
À
_fn_kwargs
Á	variables
Â	keras_api*
M

Ãtotal

Äcount
Å
_fn_kwargs
Æ	variables
Ç	keras_api*

-0
.1*

-0
.1*
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses* 
* 
* 

)0
*1*

)0
*1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses* 
* 
* 
* 

+0
,1*

+0
,1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
ð	variables
ñtrainable_variables
òregularization_losses
ô__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses* 
* 
* 
* 
C
L0
M1
N2
O3
P4
Q5
R6
J7
K8*
* 
* 
* 
* 
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

/0
01*

/0
01*
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

10
21*

10
21*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses* 
* 
* 
* 
C
\0
]1
^2
_3
`4
a5
b6
Z7
[8*
* 
* 
* 
* 
* 
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses* 
* 
* 

30
41*

30
41*
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses* 
* 
* 
* 

50
61*

50
61*
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses* 
* 
* 
* 
C
l0
m1
n2
o3
p4
q5
r6
j7
k8*
* 
* 
* 
* 
* 
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

70
81*

70
81*
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

90
:1*

90
:1*
* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses* 
* 
* 
* 
F
|0
}1
~2
3
4
5
6
z7
{8*
* 
* 
* 
* 
* 
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses* 
* 
* 

;0
<1*

;0
<1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses* 
* 
* 
* 

=0
>1*

=0
>1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses* 
* 
* 
* 
L
0
1
2
3
4
5
6
7
8*
* 
* 
* 
* 
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses* 
* 
* 

?0
@1*

?0
@1*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

A0
B1*

A0
B1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 
* 
* 
* 
L
0
1
2
3
 4
¡5
¢6
7
8*
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

±0
²1*

³	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

µ0
¶1*

·	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¹0
º1*

¼	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¾0
¿1*

Á	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ã0
Ä1*

Æ	variables*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/tcn/residual_block_0/matching_conv1D/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/tcn/residual_block_0/matching_conv1D/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_0/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_0/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_1/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_1/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_4/conv1D_0/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_4/conv1D_0/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_4/conv1D_1/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_4/conv1D_1/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_5/conv1D_0/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_5/conv1D_0/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_5/conv1D_1/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_5/conv1D_1/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_0/conv1D_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_0/conv1D_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/tcn/residual_block_0/matching_conv1D/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/tcn/residual_block_0/matching_conv1D/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_1/conv1D_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_1/conv1D_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_2/conv1D_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_2/conv1D_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_0/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_0/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_3/conv1D_1/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_3/conv1D_1/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_4/conv1D_0/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_4/conv1D_0/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_4/conv1D_1/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_4/conv1D_1/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_5/conv1D_0/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_5/conv1D_0/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tcn/residual_block_5/conv1D_1/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tcn/residual_block_5/conv1D_1/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_tcn_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ0
Õ

StatefulPartitionedCallStatefulPartitionedCallserving_default_tcn_input$tcn/residual_block_0/conv1D_0/kernel"tcn/residual_block_0/conv1D_0/bias$tcn/residual_block_0/conv1D_1/kernel"tcn/residual_block_0/conv1D_1/bias+tcn/residual_block_0/matching_conv1D/kernel)tcn/residual_block_0/matching_conv1D/bias$tcn/residual_block_1/conv1D_0/kernel"tcn/residual_block_1/conv1D_0/bias$tcn/residual_block_1/conv1D_1/kernel"tcn/residual_block_1/conv1D_1/bias$tcn/residual_block_2/conv1D_0/kernel"tcn/residual_block_2/conv1D_0/bias$tcn/residual_block_2/conv1D_1/kernel"tcn/residual_block_2/conv1D_1/bias$tcn/residual_block_3/conv1D_0/kernel"tcn/residual_block_3/conv1D_0/bias$tcn/residual_block_3/conv1D_1/kernel"tcn/residual_block_3/conv1D_1/bias$tcn/residual_block_4/conv1D_0/kernel"tcn/residual_block_4/conv1D_0/bias$tcn/residual_block_4/conv1D_1/kernel"tcn/residual_block_4/conv1D_1/bias$tcn/residual_block_5/conv1D_0/kernel"tcn/residual_block_5/conv1D_0/bias$tcn/residual_block_5/conv1D_1/kernel"tcn/residual_block_5/conv1D_1/biasdense/kernel
dense/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_75613
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
.
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp8tcn/residual_block_0/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_0/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_0/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_0/conv1D_1/bias/Read/ReadVariableOp?tcn/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOp=tcn/residual_block_0/matching_conv1D/bias/Read/ReadVariableOp8tcn/residual_block_1/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_1/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_1/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_1/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_2/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_2/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_2/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_2/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_3/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_3/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_3/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_3/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_4/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_4/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_4/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_4/conv1D_1/bias/Read/ReadVariableOp8tcn/residual_block_5/conv1D_0/kernel/Read/ReadVariableOp6tcn/residual_block_5/conv1D_0/bias/Read/ReadVariableOp8tcn/residual_block_5/conv1D_1/kernel/Read/ReadVariableOp6tcn/residual_block_5/conv1D_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpFAdam/tcn/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpDAdam/tcn/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_1/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_4/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_4/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_4/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_4/conv1D_1/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_5/conv1D_0/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_5/conv1D_0/bias/m/Read/ReadVariableOp?Adam/tcn/residual_block_5/conv1D_1/kernel/m/Read/ReadVariableOp=Adam/tcn/residual_block_5/conv1D_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpFAdam/tcn/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpDAdam/tcn/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_3/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_3/conv1D_1/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_4/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_4/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_4/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_4/conv1D_1/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_5/conv1D_0/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_5/conv1D_0/bias/v/Read/ReadVariableOp?Adam/tcn/residual_block_5/conv1D_1/kernel/v/Read/ReadVariableOp=Adam/tcn/residual_block_5/conv1D_1/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_77162
Ù
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate$tcn/residual_block_0/conv1D_0/kernel"tcn/residual_block_0/conv1D_0/bias$tcn/residual_block_0/conv1D_1/kernel"tcn/residual_block_0/conv1D_1/bias+tcn/residual_block_0/matching_conv1D/kernel)tcn/residual_block_0/matching_conv1D/bias$tcn/residual_block_1/conv1D_0/kernel"tcn/residual_block_1/conv1D_0/bias$tcn/residual_block_1/conv1D_1/kernel"tcn/residual_block_1/conv1D_1/bias$tcn/residual_block_2/conv1D_0/kernel"tcn/residual_block_2/conv1D_0/bias$tcn/residual_block_2/conv1D_1/kernel"tcn/residual_block_2/conv1D_1/bias$tcn/residual_block_3/conv1D_0/kernel"tcn/residual_block_3/conv1D_0/bias$tcn/residual_block_3/conv1D_1/kernel"tcn/residual_block_3/conv1D_1/bias$tcn/residual_block_4/conv1D_0/kernel"tcn/residual_block_4/conv1D_0/bias$tcn/residual_block_4/conv1D_1/kernel"tcn/residual_block_4/conv1D_1/bias$tcn/residual_block_5/conv1D_0/kernel"tcn/residual_block_5/conv1D_0/bias$tcn/residual_block_5/conv1D_1/kernel"tcn/residual_block_5/conv1D_1/biastotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4Adam/dense/kernel/mAdam/dense/bias/m+Adam/tcn/residual_block_0/conv1D_0/kernel/m)Adam/tcn/residual_block_0/conv1D_0/bias/m+Adam/tcn/residual_block_0/conv1D_1/kernel/m)Adam/tcn/residual_block_0/conv1D_1/bias/m2Adam/tcn/residual_block_0/matching_conv1D/kernel/m0Adam/tcn/residual_block_0/matching_conv1D/bias/m+Adam/tcn/residual_block_1/conv1D_0/kernel/m)Adam/tcn/residual_block_1/conv1D_0/bias/m+Adam/tcn/residual_block_1/conv1D_1/kernel/m)Adam/tcn/residual_block_1/conv1D_1/bias/m+Adam/tcn/residual_block_2/conv1D_0/kernel/m)Adam/tcn/residual_block_2/conv1D_0/bias/m+Adam/tcn/residual_block_2/conv1D_1/kernel/m)Adam/tcn/residual_block_2/conv1D_1/bias/m+Adam/tcn/residual_block_3/conv1D_0/kernel/m)Adam/tcn/residual_block_3/conv1D_0/bias/m+Adam/tcn/residual_block_3/conv1D_1/kernel/m)Adam/tcn/residual_block_3/conv1D_1/bias/m+Adam/tcn/residual_block_4/conv1D_0/kernel/m)Adam/tcn/residual_block_4/conv1D_0/bias/m+Adam/tcn/residual_block_4/conv1D_1/kernel/m)Adam/tcn/residual_block_4/conv1D_1/bias/m+Adam/tcn/residual_block_5/conv1D_0/kernel/m)Adam/tcn/residual_block_5/conv1D_0/bias/m+Adam/tcn/residual_block_5/conv1D_1/kernel/m)Adam/tcn/residual_block_5/conv1D_1/bias/mAdam/dense/kernel/vAdam/dense/bias/v+Adam/tcn/residual_block_0/conv1D_0/kernel/v)Adam/tcn/residual_block_0/conv1D_0/bias/v+Adam/tcn/residual_block_0/conv1D_1/kernel/v)Adam/tcn/residual_block_0/conv1D_1/bias/v2Adam/tcn/residual_block_0/matching_conv1D/kernel/v0Adam/tcn/residual_block_0/matching_conv1D/bias/v+Adam/tcn/residual_block_1/conv1D_0/kernel/v)Adam/tcn/residual_block_1/conv1D_0/bias/v+Adam/tcn/residual_block_1/conv1D_1/kernel/v)Adam/tcn/residual_block_1/conv1D_1/bias/v+Adam/tcn/residual_block_2/conv1D_0/kernel/v)Adam/tcn/residual_block_2/conv1D_0/bias/v+Adam/tcn/residual_block_2/conv1D_1/kernel/v)Adam/tcn/residual_block_2/conv1D_1/bias/v+Adam/tcn/residual_block_3/conv1D_0/kernel/v)Adam/tcn/residual_block_3/conv1D_0/bias/v+Adam/tcn/residual_block_3/conv1D_1/kernel/v)Adam/tcn/residual_block_3/conv1D_1/bias/v+Adam/tcn/residual_block_4/conv1D_0/kernel/v)Adam/tcn/residual_block_4/conv1D_0/bias/v+Adam/tcn/residual_block_4/conv1D_1/kernel/v)Adam/tcn/residual_block_4/conv1D_1/bias/v+Adam/tcn/residual_block_5/conv1D_0/kernel/v)Adam/tcn/residual_block_5/conv1D_0/bias/v+Adam/tcn/residual_block_5/conv1D_1/kernel/v)Adam/tcn/residual_block_5/conv1D_1/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_77469Õù)
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76829

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76712

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73136v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73016

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76539

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_72938v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76735

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73148v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
±
>__inference_tcn_layer_call_and_return_conditional_losses_73598

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:@b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_1_biasadd_readvariableop_resource:@
identity¢0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Æ
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@û
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@Æ
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÃ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0Ô
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
Â
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ´
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ä
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ¾
residual_block_4/conv1D_0/PadPad1residual_block_3/Act_Res_Block/Relu:activations:0/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_0/Pad:output:0Dresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_0/Conv1DConv2D4residual_block_4/conv1D_0/Conv1D/ExpandDims:output:06residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_0/BiasAddBiasAdd8residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_0/ReluRelu*residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_4/SDropout_0/IdentityIdentity0residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        º
residual_block_4/conv1D_1/PadPad-residual_block_4/SDropout_0/Identity:output:0/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_1/Pad:output:0Dresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_1/Conv1DConv2D4residual_block_4/conv1D_1/Conv1D/ExpandDims:output:06residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_1/BiasAddBiasAdd8residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_1/ReluRelu*residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_4/SDropout_1/IdentityIdentity0residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_4/Act_Conv_Blocks/ReluRelu-residual_block_4/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_4/Add_Res/addAddV21residual_block_3/Act_Res_Block/Relu:activations:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_4/Act_Res_Block/ReluRelu residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ¾
residual_block_5/conv1D_0/PadPad1residual_block_4/Act_Res_Block/Relu:activations:0/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_0/Pad:output:0Dresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_0/Conv1DConv2D4residual_block_5/conv1D_0/Conv1D/ExpandDims:output:06residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_0/BiasAddBiasAdd8residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_0/ReluRelu*residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_5/SDropout_0/IdentityIdentity0residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               º
residual_block_5/conv1D_1/PadPad-residual_block_5/SDropout_0/Identity:output:0/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_1/Pad:output:0Dresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_1/Conv1DConv2D4residual_block_5/conv1D_1/Conv1D/ExpandDims:output:06residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_1/BiasAddBiasAdd8residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_1/ReluRelu*residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_5/SDropout_1/IdentityIdentity0residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_5/Act_Conv_Blocks/ReluRelu-residual_block_5/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_5/Add_Res/addAddV21residual_block_4/Act_Res_Block/Relu:activations:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_5/Act_Res_Block/ReluRelu residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Á
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¬
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_3AddV2Add_Skip_Connections/add_2:z:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_4AddV2Add_Skip_Connections/add_3:z:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         È
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_4:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73136

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_72956

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72908

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76567

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72968v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76549

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76661

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76511

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72908v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_72998

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76623

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73028v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76544

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_72956v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73088

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76618

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76646

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76577

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76814

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72986

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73226

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76707

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73118v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Î
E__inference_sequential_layer_call_and_return_conditional_losses_74406

inputs
	tcn_74347:@
	tcn_74349:@
	tcn_74351:@@
	tcn_74353:@
	tcn_74355:@
	tcn_74357:@
	tcn_74359:@@
	tcn_74361:@
	tcn_74363:@@
	tcn_74365:@
	tcn_74367:@@
	tcn_74369:@
	tcn_74371:@@
	tcn_74373:@
	tcn_74375:@@
	tcn_74377:@
	tcn_74379:@@
	tcn_74381:@
	tcn_74383:@@
	tcn_74385:@
	tcn_74387:@@
	tcn_74389:@
	tcn_74391:@@
	tcn_74393:@
	tcn_74395:@@
	tcn_74397:@
dense_74400:@
dense_74402:
identity¢dense/StatefulPartitionedCall¢tcn/StatefulPartitionedCall
tcn/StatefulPartitionedCallStatefulPartitionedCallinputs	tcn_74347	tcn_74349	tcn_74351	tcn_74353	tcn_74355	tcn_74357	tcn_74359	tcn_74361	tcn_74363	tcn_74365	tcn_74367	tcn_74369	tcn_74371	tcn_74373	tcn_74375	tcn_74377	tcn_74379	tcn_74381	tcn_74383	tcn_74385	tcn_74387	tcn_74389	tcn_74391	tcn_74393	tcn_74395	tcn_74397*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_tcn_layer_call_and_return_conditional_losses_74226ÿ
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_74400dense_74402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73662u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76730

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72968

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76796

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73226v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

#__inference_signature_wrapper_75613
	tcn_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	tcn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_72899o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
#
_user_specified_name	tcn_input
æ

*__inference_sequential_layer_call_fn_74717

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_73669o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76572

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72986v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÄÈ
É4
__inference__traced_save_77162
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_0_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_0_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_0_conv1d_1_bias_read_readvariableopJ
Fsavev2_tcn_residual_block_0_matching_conv1d_kernel_read_readvariableopH
Dsavev2_tcn_residual_block_0_matching_conv1d_bias_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_1_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_1_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_1_conv1d_1_bias_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_2_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_2_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_2_conv1d_1_bias_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_3_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_3_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_3_conv1d_1_bias_read_readvariableopC
?savev2_tcn_residual_block_4_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_4_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_4_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_4_conv1d_1_bias_read_readvariableopC
?savev2_tcn_residual_block_5_conv1d_0_kernel_read_readvariableopA
=savev2_tcn_residual_block_5_conv1d_0_bias_read_readvariableopC
?savev2_tcn_residual_block_5_conv1d_1_kernel_read_readvariableopA
=savev2_tcn_residual_block_5_conv1d_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_1_bias_m_read_readvariableopQ
Msavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_m_read_readvariableopO
Ksavev2_adam_tcn_residual_block_0_matching_conv1d_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_1_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_1_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_1_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_4_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_4_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_4_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_4_conv1d_1_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_5_conv1d_0_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_5_conv1d_0_bias_m_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_5_conv1d_1_kernel_m_read_readvariableopH
Dsavev2_adam_tcn_residual_block_5_conv1d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_0_conv1d_1_bias_v_read_readvariableopQ
Msavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_v_read_readvariableopO
Ksavev2_adam_tcn_residual_block_0_matching_conv1d_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_1_conv1d_1_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_2_conv1d_1_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_3_conv1d_1_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_4_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_4_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_4_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_4_conv1d_1_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_5_conv1d_0_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_5_conv1d_0_bias_v_read_readvariableopJ
Fsavev2_adam_tcn_residual_block_5_conv1d_1_kernel_v_read_readvariableopH
Dsavev2_adam_tcn_residual_block_5_conv1d_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Õ.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*þ-
valueô-Bñ-dB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¸
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBÐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B å2
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop?savev2_tcn_residual_block_0_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_0_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_0_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_0_conv1d_1_bias_read_readvariableopFsavev2_tcn_residual_block_0_matching_conv1d_kernel_read_readvariableopDsavev2_tcn_residual_block_0_matching_conv1d_bias_read_readvariableop?savev2_tcn_residual_block_1_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_1_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_1_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_1_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_2_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_2_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_2_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_2_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_3_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_3_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_3_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_3_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_4_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_4_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_4_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_4_conv1d_1_bias_read_readvariableop?savev2_tcn_residual_block_5_conv1d_0_kernel_read_readvariableop=savev2_tcn_residual_block_5_conv1d_0_bias_read_readvariableop?savev2_tcn_residual_block_5_conv1d_1_kernel_read_readvariableop=savev2_tcn_residual_block_5_conv1d_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_1_bias_m_read_readvariableopMsavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_m_read_readvariableopKsavev2_adam_tcn_residual_block_0_matching_conv1d_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_1_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_1_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_1_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_4_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_4_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_4_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_4_conv1d_1_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_5_conv1d_0_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_5_conv1d_0_bias_m_read_readvariableopFsavev2_adam_tcn_residual_block_5_conv1d_1_kernel_m_read_readvariableopDsavev2_adam_tcn_residual_block_5_conv1d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_0_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_0_conv1d_1_bias_v_read_readvariableopMsavev2_adam_tcn_residual_block_0_matching_conv1d_kernel_v_read_readvariableopKsavev2_adam_tcn_residual_block_0_matching_conv1d_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_1_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_1_conv1d_1_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_2_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_2_conv1d_1_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_3_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_3_conv1d_1_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_4_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_4_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_4_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_4_conv1d_1_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_5_conv1d_0_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_5_conv1d_0_bias_v_read_readvariableopFsavev2_adam_tcn_residual_block_5_conv1d_1_kernel_v_read_readvariableopDsavev2_adam_tcn_residual_block_5_conv1d_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ó
_input_shapesá
Þ: :@:: : : : : :@:@:@@:@:@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: : : : : : : : : : :@::@:@:@@:@:@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@:@:@@:@:@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:@: 	

_output_shapes
:@:(
$
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:( $
"
_output_shapes
:@@: !

_output_shapes
:@:"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

:@: -

_output_shapes
::(.$
"
_output_shapes
:@: /

_output_shapes
:@:(0$
"
_output_shapes
:@@: 1

_output_shapes
:@:(2$
"
_output_shapes
:@: 3

_output_shapes
:@:(4$
"
_output_shapes
:@@: 5

_output_shapes
:@:(6$
"
_output_shapes
:@@: 7

_output_shapes
:@:(8$
"
_output_shapes
:@@: 9

_output_shapes
:@:(:$
"
_output_shapes
:@@: ;

_output_shapes
:@:(<$
"
_output_shapes
:@@: =

_output_shapes
:@:(>$
"
_output_shapes
:@@: ?

_output_shapes
:@:(@$
"
_output_shapes
:@@: A

_output_shapes
:@:(B$
"
_output_shapes
:@@: C

_output_shapes
:@:(D$
"
_output_shapes
:@@: E

_output_shapes
:@:(F$
"
_output_shapes
:@@: G

_output_shapes
:@:$H 

_output_shapes

:@: I

_output_shapes
::(J$
"
_output_shapes
:@: K

_output_shapes
:@:(L$
"
_output_shapes
:@@: M

_output_shapes
:@:(N$
"
_output_shapes
:@: O

_output_shapes
:@:(P$
"
_output_shapes
:@@: Q

_output_shapes
:@:(R$
"
_output_shapes
:@@: S

_output_shapes
:@:(T$
"
_output_shapes
:@@: U

_output_shapes
:@:(V$
"
_output_shapes
:@@: W

_output_shapes
:@:(X$
"
_output_shapes
:@@: Y

_output_shapes
:@:(Z$
"
_output_shapes
:@@: [

_output_shapes
:@:(\$
"
_output_shapes
:@@: ]

_output_shapes
:@:(^$
"
_output_shapes
:@@: _

_output_shapes
:@:(`$
"
_output_shapes
:@@: a

_output_shapes
:@:(b$
"
_output_shapes
:@@: c

_output_shapes
:@:d

_output_shapes
: 
º

%__inference_dense_layer_call_fn_76496

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73662o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76651

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73058v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

*__inference_sequential_layer_call_fn_74526
	tcn_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCall	tcn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_74406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
#
_user_specified_name	tcn_input
÷
F
*__inference_SDropout_0_layer_call_fn_76740

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73166v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
±
>__inference_tcn_layer_call_and_return_conditional_losses_76059

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:@b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_1_biasadd_readvariableop_resource:@
identity¢0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Æ
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@û
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@Æ
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÃ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0Ô
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
Â
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ´
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ä
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       º
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ¾
residual_block_4/conv1D_0/PadPad1residual_block_3/Act_Res_Block/Relu:activations:0/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_0/Pad:output:0Dresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_0/Conv1DConv2D4residual_block_4/conv1D_0/Conv1D/ExpandDims:output:06residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_0/BiasAddBiasAdd8residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_0/ReluRelu*residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_4/SDropout_0/IdentityIdentity0residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        º
residual_block_4/conv1D_1/PadPad-residual_block_4/SDropout_0/Identity:output:0/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_1/Pad:output:0Dresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_1/Conv1DConv2D4residual_block_4/conv1D_1/Conv1D/ExpandDims:output:06residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_1/BiasAddBiasAdd8residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_1/ReluRelu*residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_4/SDropout_1/IdentityIdentity0residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_4/Act_Conv_Blocks/ReluRelu-residual_block_4/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_4/Add_Res/addAddV21residual_block_3/Act_Res_Block/Relu:activations:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_4/Act_Res_Block/ReluRelu residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ¾
residual_block_5/conv1D_0/PadPad1residual_block_4/Act_Res_Block/Relu:activations:0/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_0/Pad:output:0Dresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_0/Conv1DConv2D4residual_block_5/conv1D_0/Conv1D/ExpandDims:output:06residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_0/BiasAddBiasAdd8residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_0/ReluRelu*residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_5/SDropout_0/IdentityIdentity0residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               º
residual_block_5/conv1D_1/PadPad-residual_block_5/SDropout_0/Identity:output:0/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_1/Pad:output:0Dresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_1/Conv1DConv2D4residual_block_5/conv1D_1/Conv1D/ExpandDims:output:06residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_1/BiasAddBiasAdd8residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_1/ReluRelu*residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
$residual_block_5/SDropout_1/IdentityIdentity0residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%residual_block_5/Act_Conv_Blocks/ReluRelu-residual_block_5/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_5/Add_Res/addAddV21residual_block_4/Act_Res_Block/Relu:activations:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_5/Act_Res_Block/ReluRelu residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Á
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¬
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_3AddV2Add_Skip_Connections/add_2:z:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_4AddV2Add_Skip_Connections/add_3:z:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         È
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_4:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76590

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76689

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ÂI
!__inference__traced_restore_77469
file_prefix/
assignvariableop_dense_kernel:@+
assignvariableop_1_dense_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: M
7assignvariableop_7_tcn_residual_block_0_conv1d_0_kernel:@C
5assignvariableop_8_tcn_residual_block_0_conv1d_0_bias:@M
7assignvariableop_9_tcn_residual_block_0_conv1d_1_kernel:@@D
6assignvariableop_10_tcn_residual_block_0_conv1d_1_bias:@U
?assignvariableop_11_tcn_residual_block_0_matching_conv1d_kernel:@K
=assignvariableop_12_tcn_residual_block_0_matching_conv1d_bias:@N
8assignvariableop_13_tcn_residual_block_1_conv1d_0_kernel:@@D
6assignvariableop_14_tcn_residual_block_1_conv1d_0_bias:@N
8assignvariableop_15_tcn_residual_block_1_conv1d_1_kernel:@@D
6assignvariableop_16_tcn_residual_block_1_conv1d_1_bias:@N
8assignvariableop_17_tcn_residual_block_2_conv1d_0_kernel:@@D
6assignvariableop_18_tcn_residual_block_2_conv1d_0_bias:@N
8assignvariableop_19_tcn_residual_block_2_conv1d_1_kernel:@@D
6assignvariableop_20_tcn_residual_block_2_conv1d_1_bias:@N
8assignvariableop_21_tcn_residual_block_3_conv1d_0_kernel:@@D
6assignvariableop_22_tcn_residual_block_3_conv1d_0_bias:@N
8assignvariableop_23_tcn_residual_block_3_conv1d_1_kernel:@@D
6assignvariableop_24_tcn_residual_block_3_conv1d_1_bias:@N
8assignvariableop_25_tcn_residual_block_4_conv1d_0_kernel:@@D
6assignvariableop_26_tcn_residual_block_4_conv1d_0_bias:@N
8assignvariableop_27_tcn_residual_block_4_conv1d_1_kernel:@@D
6assignvariableop_28_tcn_residual_block_4_conv1d_1_bias:@N
8assignvariableop_29_tcn_residual_block_5_conv1d_0_kernel:@@D
6assignvariableop_30_tcn_residual_block_5_conv1d_0_bias:@N
8assignvariableop_31_tcn_residual_block_5_conv1d_1_kernel:@@D
6assignvariableop_32_tcn_residual_block_5_conv1d_1_bias:@#
assignvariableop_33_total: #
assignvariableop_34_count: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: %
assignvariableop_37_total_2: %
assignvariableop_38_count_2: %
assignvariableop_39_total_3: %
assignvariableop_40_count_3: %
assignvariableop_41_total_4: %
assignvariableop_42_count_4: 9
'assignvariableop_43_adam_dense_kernel_m:@3
%assignvariableop_44_adam_dense_bias_m:U
?assignvariableop_45_adam_tcn_residual_block_0_conv1d_0_kernel_m:@K
=assignvariableop_46_adam_tcn_residual_block_0_conv1d_0_bias_m:@U
?assignvariableop_47_adam_tcn_residual_block_0_conv1d_1_kernel_m:@@K
=assignvariableop_48_adam_tcn_residual_block_0_conv1d_1_bias_m:@\
Fassignvariableop_49_adam_tcn_residual_block_0_matching_conv1d_kernel_m:@R
Dassignvariableop_50_adam_tcn_residual_block_0_matching_conv1d_bias_m:@U
?assignvariableop_51_adam_tcn_residual_block_1_conv1d_0_kernel_m:@@K
=assignvariableop_52_adam_tcn_residual_block_1_conv1d_0_bias_m:@U
?assignvariableop_53_adam_tcn_residual_block_1_conv1d_1_kernel_m:@@K
=assignvariableop_54_adam_tcn_residual_block_1_conv1d_1_bias_m:@U
?assignvariableop_55_adam_tcn_residual_block_2_conv1d_0_kernel_m:@@K
=assignvariableop_56_adam_tcn_residual_block_2_conv1d_0_bias_m:@U
?assignvariableop_57_adam_tcn_residual_block_2_conv1d_1_kernel_m:@@K
=assignvariableop_58_adam_tcn_residual_block_2_conv1d_1_bias_m:@U
?assignvariableop_59_adam_tcn_residual_block_3_conv1d_0_kernel_m:@@K
=assignvariableop_60_adam_tcn_residual_block_3_conv1d_0_bias_m:@U
?assignvariableop_61_adam_tcn_residual_block_3_conv1d_1_kernel_m:@@K
=assignvariableop_62_adam_tcn_residual_block_3_conv1d_1_bias_m:@U
?assignvariableop_63_adam_tcn_residual_block_4_conv1d_0_kernel_m:@@K
=assignvariableop_64_adam_tcn_residual_block_4_conv1d_0_bias_m:@U
?assignvariableop_65_adam_tcn_residual_block_4_conv1d_1_kernel_m:@@K
=assignvariableop_66_adam_tcn_residual_block_4_conv1d_1_bias_m:@U
?assignvariableop_67_adam_tcn_residual_block_5_conv1d_0_kernel_m:@@K
=assignvariableop_68_adam_tcn_residual_block_5_conv1d_0_bias_m:@U
?assignvariableop_69_adam_tcn_residual_block_5_conv1d_1_kernel_m:@@K
=assignvariableop_70_adam_tcn_residual_block_5_conv1d_1_bias_m:@9
'assignvariableop_71_adam_dense_kernel_v:@3
%assignvariableop_72_adam_dense_bias_v:U
?assignvariableop_73_adam_tcn_residual_block_0_conv1d_0_kernel_v:@K
=assignvariableop_74_adam_tcn_residual_block_0_conv1d_0_bias_v:@U
?assignvariableop_75_adam_tcn_residual_block_0_conv1d_1_kernel_v:@@K
=assignvariableop_76_adam_tcn_residual_block_0_conv1d_1_bias_v:@\
Fassignvariableop_77_adam_tcn_residual_block_0_matching_conv1d_kernel_v:@R
Dassignvariableop_78_adam_tcn_residual_block_0_matching_conv1d_bias_v:@U
?assignvariableop_79_adam_tcn_residual_block_1_conv1d_0_kernel_v:@@K
=assignvariableop_80_adam_tcn_residual_block_1_conv1d_0_bias_v:@U
?assignvariableop_81_adam_tcn_residual_block_1_conv1d_1_kernel_v:@@K
=assignvariableop_82_adam_tcn_residual_block_1_conv1d_1_bias_v:@U
?assignvariableop_83_adam_tcn_residual_block_2_conv1d_0_kernel_v:@@K
=assignvariableop_84_adam_tcn_residual_block_2_conv1d_0_bias_v:@U
?assignvariableop_85_adam_tcn_residual_block_2_conv1d_1_kernel_v:@@K
=assignvariableop_86_adam_tcn_residual_block_2_conv1d_1_bias_v:@U
?assignvariableop_87_adam_tcn_residual_block_3_conv1d_0_kernel_v:@@K
=assignvariableop_88_adam_tcn_residual_block_3_conv1d_0_bias_v:@U
?assignvariableop_89_adam_tcn_residual_block_3_conv1d_1_kernel_v:@@K
=assignvariableop_90_adam_tcn_residual_block_3_conv1d_1_bias_v:@U
?assignvariableop_91_adam_tcn_residual_block_4_conv1d_0_kernel_v:@@K
=assignvariableop_92_adam_tcn_residual_block_4_conv1d_0_bias_v:@U
?assignvariableop_93_adam_tcn_residual_block_4_conv1d_1_kernel_v:@@K
=assignvariableop_94_adam_tcn_residual_block_4_conv1d_1_bias_v:@U
?assignvariableop_95_adam_tcn_residual_block_5_conv1d_0_kernel_v:@@K
=assignvariableop_96_adam_tcn_residual_block_5_conv1d_0_bias_v:@U
?assignvariableop_97_adam_tcn_residual_block_5_conv1d_1_kernel_v:@@K
=assignvariableop_98_adam_tcn_residual_block_5_conv1d_1_bias_v:@
identity_100¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98Ø.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*þ-
valueô-Bñ-dB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH»
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBÐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_7AssignVariableOp7assignvariableop_7_tcn_residual_block_0_conv1d_0_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_8AssignVariableOp5assignvariableop_8_tcn_residual_block_0_conv1d_0_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_9AssignVariableOp7assignvariableop_9_tcn_residual_block_0_conv1d_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_10AssignVariableOp6assignvariableop_10_tcn_residual_block_0_conv1d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_11AssignVariableOp?assignvariableop_11_tcn_residual_block_0_matching_conv1d_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_12AssignVariableOp=assignvariableop_12_tcn_residual_block_0_matching_conv1d_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_13AssignVariableOp8assignvariableop_13_tcn_residual_block_1_conv1d_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp6assignvariableop_14_tcn_residual_block_1_conv1d_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_15AssignVariableOp8assignvariableop_15_tcn_residual_block_1_conv1d_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_16AssignVariableOp6assignvariableop_16_tcn_residual_block_1_conv1d_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_17AssignVariableOp8assignvariableop_17_tcn_residual_block_2_conv1d_0_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_18AssignVariableOp6assignvariableop_18_tcn_residual_block_2_conv1d_0_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_19AssignVariableOp8assignvariableop_19_tcn_residual_block_2_conv1d_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_20AssignVariableOp6assignvariableop_20_tcn_residual_block_2_conv1d_1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_21AssignVariableOp8assignvariableop_21_tcn_residual_block_3_conv1d_0_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_22AssignVariableOp6assignvariableop_22_tcn_residual_block_3_conv1d_0_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_23AssignVariableOp8assignvariableop_23_tcn_residual_block_3_conv1d_1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_tcn_residual_block_3_conv1d_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_25AssignVariableOp8assignvariableop_25_tcn_residual_block_4_conv1d_0_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_tcn_residual_block_4_conv1d_0_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_27AssignVariableOp8assignvariableop_27_tcn_residual_block_4_conv1d_1_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_28AssignVariableOp6assignvariableop_28_tcn_residual_block_4_conv1d_1_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_29AssignVariableOp8assignvariableop_29_tcn_residual_block_5_conv1d_0_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_30AssignVariableOp6assignvariableop_30_tcn_residual_block_5_conv1d_0_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_31AssignVariableOp8assignvariableop_31_tcn_residual_block_5_conv1d_1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_32AssignVariableOp6assignvariableop_32_tcn_residual_block_5_conv1d_1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_3Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_4Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_4Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_dense_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_tcn_residual_block_0_conv1d_0_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_46AssignVariableOp=assignvariableop_46_adam_tcn_residual_block_0_conv1d_0_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_47AssignVariableOp?assignvariableop_47_adam_tcn_residual_block_0_conv1d_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_48AssignVariableOp=assignvariableop_48_adam_tcn_residual_block_0_conv1d_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_49AssignVariableOpFassignvariableop_49_adam_tcn_residual_block_0_matching_conv1d_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_50AssignVariableOpDassignvariableop_50_adam_tcn_residual_block_0_matching_conv1d_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_51AssignVariableOp?assignvariableop_51_adam_tcn_residual_block_1_conv1d_0_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_52AssignVariableOp=assignvariableop_52_adam_tcn_residual_block_1_conv1d_0_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_53AssignVariableOp?assignvariableop_53_adam_tcn_residual_block_1_conv1d_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_54AssignVariableOp=assignvariableop_54_adam_tcn_residual_block_1_conv1d_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_55AssignVariableOp?assignvariableop_55_adam_tcn_residual_block_2_conv1d_0_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_56AssignVariableOp=assignvariableop_56_adam_tcn_residual_block_2_conv1d_0_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_57AssignVariableOp?assignvariableop_57_adam_tcn_residual_block_2_conv1d_1_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_58AssignVariableOp=assignvariableop_58_adam_tcn_residual_block_2_conv1d_1_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_59AssignVariableOp?assignvariableop_59_adam_tcn_residual_block_3_conv1d_0_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_60AssignVariableOp=assignvariableop_60_adam_tcn_residual_block_3_conv1d_0_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_61AssignVariableOp?assignvariableop_61_adam_tcn_residual_block_3_conv1d_1_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_62AssignVariableOp=assignvariableop_62_adam_tcn_residual_block_3_conv1d_1_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_63AssignVariableOp?assignvariableop_63_adam_tcn_residual_block_4_conv1d_0_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_64AssignVariableOp=assignvariableop_64_adam_tcn_residual_block_4_conv1d_0_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_tcn_residual_block_4_conv1d_1_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_66AssignVariableOp=assignvariableop_66_adam_tcn_residual_block_4_conv1d_1_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_67AssignVariableOp?assignvariableop_67_adam_tcn_residual_block_5_conv1d_0_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_68AssignVariableOp=assignvariableop_68_adam_tcn_residual_block_5_conv1d_0_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_69AssignVariableOp?assignvariableop_69_adam_tcn_residual_block_5_conv1d_1_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_70AssignVariableOp=assignvariableop_70_adam_tcn_residual_block_5_conv1d_1_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_dense_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_dense_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_73AssignVariableOp?assignvariableop_73_adam_tcn_residual_block_0_conv1d_0_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_74AssignVariableOp=assignvariableop_74_adam_tcn_residual_block_0_conv1d_0_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_75AssignVariableOp?assignvariableop_75_adam_tcn_residual_block_0_conv1d_1_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_76AssignVariableOp=assignvariableop_76_adam_tcn_residual_block_0_conv1d_1_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_77AssignVariableOpFassignvariableop_77_adam_tcn_residual_block_0_matching_conv1d_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_78AssignVariableOpDassignvariableop_78_adam_tcn_residual_block_0_matching_conv1d_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_79AssignVariableOp?assignvariableop_79_adam_tcn_residual_block_1_conv1d_0_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_80AssignVariableOp=assignvariableop_80_adam_tcn_residual_block_1_conv1d_0_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_81AssignVariableOp?assignvariableop_81_adam_tcn_residual_block_1_conv1d_1_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_82AssignVariableOp=assignvariableop_82_adam_tcn_residual_block_1_conv1d_1_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_83AssignVariableOp?assignvariableop_83_adam_tcn_residual_block_2_conv1d_0_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_84AssignVariableOp=assignvariableop_84_adam_tcn_residual_block_2_conv1d_0_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_85AssignVariableOp?assignvariableop_85_adam_tcn_residual_block_2_conv1d_1_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_86AssignVariableOp=assignvariableop_86_adam_tcn_residual_block_2_conv1d_1_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_87AssignVariableOp?assignvariableop_87_adam_tcn_residual_block_3_conv1d_0_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_88AssignVariableOp=assignvariableop_88_adam_tcn_residual_block_3_conv1d_0_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_89AssignVariableOp?assignvariableop_89_adam_tcn_residual_block_3_conv1d_1_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_90AssignVariableOp=assignvariableop_90_adam_tcn_residual_block_3_conv1d_1_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_91AssignVariableOp?assignvariableop_91_adam_tcn_residual_block_4_conv1d_0_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_92AssignVariableOp=assignvariableop_92_adam_tcn_residual_block_4_conv1d_0_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_93AssignVariableOp?assignvariableop_93_adam_tcn_residual_block_4_conv1d_1_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_94AssignVariableOp=assignvariableop_94_adam_tcn_residual_block_4_conv1d_1_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_95AssignVariableOp?assignvariableop_95_adam_tcn_residual_block_5_conv1d_0_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_96AssignVariableOp=assignvariableop_96_adam_tcn_residual_block_5_conv1d_0_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_97AssignVariableOp?assignvariableop_97_adam_tcn_residual_block_5_conv1d_1_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_98AssignVariableOp=assignvariableop_98_adam_tcn_residual_block_5_conv1d_1_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ñ
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: X
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: ¾
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 "%
identity_100Identity_100:output:0*Ý
_input_shapesË
È: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73178

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73058

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76702

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã	
ñ
@__inference_dense_layer_call_and_return_conditional_losses_73662

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72926

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73106

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76773

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76521

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73046

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Î
E__inference_sequential_layer_call_and_return_conditional_losses_73669

inputs
	tcn_73599:@
	tcn_73601:@
	tcn_73603:@@
	tcn_73605:@
	tcn_73607:@
	tcn_73609:@
	tcn_73611:@@
	tcn_73613:@
	tcn_73615:@@
	tcn_73617:@
	tcn_73619:@@
	tcn_73621:@
	tcn_73623:@@
	tcn_73625:@
	tcn_73627:@@
	tcn_73629:@
	tcn_73631:@@
	tcn_73633:@
	tcn_73635:@@
	tcn_73637:@
	tcn_73639:@@
	tcn_73641:@
	tcn_73643:@@
	tcn_73645:@
	tcn_73647:@@
	tcn_73649:@
dense_73663:@
dense_73665:
identity¢dense/StatefulPartitionedCall¢tcn/StatefulPartitionedCall
tcn/StatefulPartitionedCallStatefulPartitionedCallinputs	tcn_73599	tcn_73601	tcn_73603	tcn_73605	tcn_73607	tcn_73609	tcn_73611	tcn_73613	tcn_73615	tcn_73617	tcn_73619	tcn_73621	tcn_73623	tcn_73625	tcn_73627	tcn_73629	tcn_73631	tcn_73633	tcn_73635	tcn_73637	tcn_73639	tcn_73641	tcn_73643	tcn_73645	tcn_73647	tcn_73649*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_tcn_layer_call_and_return_conditional_losses_73598ÿ
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_73663dense_73665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73662u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76628

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73046v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76679

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73088v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76605

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76786

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73118

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73256

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã	
ñ
@__inference_dense_layer_call_and_return_conditional_losses_76506

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Û
#__inference_tcn_layer_call_fn_75727

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_tcn_layer_call_and_return_conditional_losses_74226o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ì
Ñ
E__inference_sequential_layer_call_and_return_conditional_losses_74650
	tcn_input
	tcn_74591:@
	tcn_74593:@
	tcn_74595:@@
	tcn_74597:@
	tcn_74599:@
	tcn_74601:@
	tcn_74603:@@
	tcn_74605:@
	tcn_74607:@@
	tcn_74609:@
	tcn_74611:@@
	tcn_74613:@
	tcn_74615:@@
	tcn_74617:@
	tcn_74619:@@
	tcn_74621:@
	tcn_74623:@@
	tcn_74625:@
	tcn_74627:@@
	tcn_74629:@
	tcn_74631:@@
	tcn_74633:@
	tcn_74635:@@
	tcn_74637:@
	tcn_74639:@@
	tcn_74641:@
dense_74644:@
dense_74646:
identity¢dense/StatefulPartitionedCall¢tcn/StatefulPartitionedCall
tcn/StatefulPartitionedCallStatefulPartitionedCall	tcn_input	tcn_74591	tcn_74593	tcn_74595	tcn_74597	tcn_74599	tcn_74601	tcn_74603	tcn_74605	tcn_74607	tcn_74609	tcn_74611	tcn_74613	tcn_74615	tcn_74617	tcn_74619	tcn_74621	tcn_74623	tcn_74625	tcn_74627	tcn_74629	tcn_74631	tcn_74633	tcn_74635	tcn_74637	tcn_74639	tcn_74641*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_tcn_layer_call_and_return_conditional_losses_74226ÿ
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_74644dense_74646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73662u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
#
_user_specified_name	tcn_input
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76801

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73208

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73196

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76674

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76745

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76842

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73166

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76717

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Û
#__inference_tcn_layer_call_fn_75670

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_tcn_layer_call_and_return_conditional_losses_73598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ÊÈ
ø$
 __inference__wrapped_model_72899
	tcn_inputj
Tsequential_tcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@V
Hsequential_tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:@q
[sequential_tcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@]
Osequential_tcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_4_conv1d_0_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_4_conv1d_1_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_5_conv1d_0_biasadd_readvariableop_resource:@j
Tsequential_tcn_residual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@V
Hsequential_tcn_residual_block_5_conv1d_1_biasadd_readvariableop_resource:@A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢Fsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Rsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢?sequential/tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢Ksequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
5sequential/tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ´
,sequential/tcn/residual_block_0/conv1D_0/PadPad	tcn_input>sequential/tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
>sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims5sequential/tcn/residual_block_0/conv1D_0/Pad:output:0Gsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ä
Ksequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0
@sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@¨
/sequential/tcn/residual_block_0/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿÄ
?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ü
0sequential/tcn/residual_block_0/conv1D_0/BiasAddBiasAdd@sequential/tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0Gsequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_0/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_0/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ç
,sequential/tcn/residual_block_0/conv1D_1/PadPad<sequential/tcn/residual_block_0/SDropout_0/Identity:output:0>sequential/tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims5sequential/tcn/residual_block_0/conv1D_1/Pad:output:0Gsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@ä
Ksequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_0/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿÄ
?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ü
0sequential/tcn/residual_block_0/conv1D_1/BiasAddBiasAdd@sequential/tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0Gsequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_0/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_0/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@°
4sequential/tcn/residual_block_0/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
Esequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿä
Asequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDims	tcn_inputNsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0ò
Rsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp[sequential_tcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0
Gsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : °
Csequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsZsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Psequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@¼
6sequential/tcn/residual_block_0/matching_conv1D/Conv1DConv2DJsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Lsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
à
>sequential/tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze?sequential/tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿÒ
Fsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpOsequential_tcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
7sequential/tcn/residual_block_0/matching_conv1D/BiasAddBiasAddGsequential/tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Nsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ð
+sequential/tcn/residual_block_0/Add_Res/addAddV2@sequential/tcn/residual_block_0/matching_conv1D/BiasAdd:output:0Bsequential/tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¡
2sequential/tcn/residual_block_0/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ë
,sequential/tcn/residual_block_1/conv1D_0/PadPad@sequential/tcn/residual_block_0/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@
=sequential/tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4·
^sequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_1/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_1/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_1/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_1/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_1/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ç
,sequential/tcn/residual_block_1/conv1D_1/PadPad<sequential/tcn/residual_block_1/SDropout_0/Identity:output:0>sequential/tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@
=sequential/tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4·
^sequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_1/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_1/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_1/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_1/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_1/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@°
4sequential/tcn/residual_block_1/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ð
+sequential/tcn/residual_block_1/Add_Res/addAddV2@sequential/tcn/residual_block_0/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¡
2sequential/tcn/residual_block_1/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ë
,sequential/tcn/residual_block_2/conv1D_0/PadPad@sequential/tcn/residual_block_1/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@
=sequential/tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8·
^sequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_2/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_2/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_2/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_2/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_2/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ç
,sequential/tcn/residual_block_2/conv1D_1/PadPad<sequential/tcn/residual_block_2/SDropout_0/Identity:output:0>sequential/tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@
=sequential/tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8·
^sequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_2/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_2/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_2/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_2/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_2/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@°
4sequential/tcn/residual_block_2/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ð
+sequential/tcn/residual_block_2/Add_Res/addAddV2@sequential/tcn/residual_block_1/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¡
2sequential/tcn/residual_block_2/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ë
,sequential/tcn/residual_block_3/conv1D_0/PadPad@sequential/tcn/residual_block_2/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
=sequential/tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@·
^sequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_3/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_3/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_3/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_3/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_3/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ç
,sequential/tcn/residual_block_3/conv1D_1/PadPad<sequential/tcn/residual_block_3/SDropout_0/Identity:output:0>sequential/tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
=sequential/tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@·
^sequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_3/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_3/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_3/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_3/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_3/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@°
4sequential/tcn/residual_block_3/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ð
+sequential/tcn/residual_block_3/Add_Res/addAddV2@sequential/tcn/residual_block_2/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¡
2sequential/tcn/residual_block_3/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ë
,sequential/tcn/residual_block_4/conv1D_0/PadPad@sequential/tcn/residual_block_3/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@
=sequential/tcn/residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P·
^sequential/tcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_4/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_4/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_4/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_4/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_4/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ç
,sequential/tcn/residual_block_4/conv1D_1/PadPad<sequential/tcn/residual_block_4/SDropout_0/Identity:output:0>sequential/tcn/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@
=sequential/tcn/residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/tcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P·
^sequential/tcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¯
Vsequential/tcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
Jsequential/tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ä
>sequential/tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_4/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_4/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ì
>sequential/tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_4/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_4/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_4/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@°
4sequential/tcn/residual_block_4/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_4/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ð
+sequential/tcn/residual_block_4/Add_Res/addAddV2@sequential/tcn/residual_block_3/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¡
2sequential/tcn/residual_block_4/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ë
,sequential/tcn/residual_block_5/conv1D_0/PadPad@sequential/tcn/residual_block_4/Act_Res_Block/Relu:activations:0>sequential/tcn/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@
=sequential/tcn/residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: ¦
\sequential/tcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p·
^sequential/tcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¯
Vsequential/tcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
Jsequential/tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:  
Gsequential/tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ä
>sequential/tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_5/conv1D_0/Pad:output:0Ssequential/tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_5/conv1D_0/Conv1DConv2DCsequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential/tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ì
>sequential/tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_5/conv1D_0/BiasAddBiasAddGsequential/tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_5/Act_Conv1D_0/ReluRelu9sequential/tcn/residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_5/SDropout_0/IdentityIdentity?sequential/tcn/residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
5sequential/tcn/residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ç
,sequential/tcn/residual_block_5/conv1D_1/PadPad<sequential/tcn/residual_block_5/SDropout_0/Identity:output:0>sequential/tcn/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@
=sequential/tcn/residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: ¦
\sequential/tcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p·
^sequential/tcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ²
Ysequential/tcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¯
Vsequential/tcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
Jsequential/tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:  
Gsequential/tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ä
>sequential/tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND5sequential/tcn/residual_block_5/conv1D_1/Pad:output:0Ssequential/tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Psequential/tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
:sequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDimsGsequential/tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:0Gsequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Ksequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_tcn_residual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
@sequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
<sequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsSsequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Isequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¨
/sequential/tcn/residual_block_5/conv1D_1/Conv1DConv2DCsequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims:output:0Esequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
Ò
7sequential/tcn/residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze8sequential/tcn/residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
Jsequential/tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential/tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ì
>sequential/tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND@sequential/tcn/residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Ssequential/tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Msequential/tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ä
?sequential/tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpHsequential_tcn_residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
0sequential/tcn/residual_block_5/conv1D_1/BiasAddBiasAddGsequential/tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:0Gsequential/tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ª
1sequential/tcn/residual_block_5/Act_Conv1D_1/ReluRelu9sequential/tcn/residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¶
3sequential/tcn/residual_block_5/SDropout_1/IdentityIdentity?sequential/tcn/residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@°
4sequential/tcn/residual_block_5/Act_Conv_Blocks/ReluRelu<sequential/tcn/residual_block_5/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ð
+sequential/tcn/residual_block_5/Add_Res/addAddV2@sequential/tcn/residual_block_4/Act_Res_Block/Relu:activations:0Bsequential/tcn/residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¡
2sequential/tcn/residual_block_5/Act_Res_Block/ReluRelu/sequential/tcn/residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@î
'sequential/tcn/Add_Skip_Connections/addAddV2Bsequential/tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0Bsequential/tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ù
)sequential/tcn/Add_Skip_Connections/add_1AddV2+sequential/tcn/Add_Skip_Connections/add:z:0Bsequential/tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Û
)sequential/tcn/Add_Skip_Connections/add_2AddV2-sequential/tcn/Add_Skip_Connections/add_1:z:0Bsequential/tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Û
)sequential/tcn/Add_Skip_Connections/add_3AddV2-sequential/tcn/Add_Skip_Connections/add_2:z:0Bsequential/tcn/residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Û
)sequential/tcn/Add_Skip_Connections/add_4AddV2-sequential/tcn/Add_Skip_Connections/add_3:z:0Bsequential/tcn/residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
/sequential/tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    
1sequential/tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
1sequential/tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
)sequential/tcn/Slice_Output/strided_sliceStridedSlice-sequential/tcn/Add_Skip_Connections/add_4:z:08sequential/tcn/Slice_Output/strided_slice/stack:output:0:sequential/tcn/Slice_Output/strided_slice/stack_1:output:0:sequential/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_mask
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0·
sequential/dense/MatMulMatMul2sequential/tcn/Slice_Output/strided_slice:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!sequential/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp@^sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpG^sequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpS^sequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@^sequential/tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOpL^sequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2
?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2
Fsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpFsequential/tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2¨
Rsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpRsequential/tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2
?sequential/tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp?sequential/tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2
Ksequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpKsequential/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
#
_user_specified_name	tcn_input
÷
F
*__inference_SDropout_0_layer_call_fn_76684

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73106v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_72938

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
Ñ
E__inference_sequential_layer_call_and_return_conditional_losses_74588
	tcn_input
	tcn_74529:@
	tcn_74531:@
	tcn_74533:@@
	tcn_74535:@
	tcn_74537:@
	tcn_74539:@
	tcn_74541:@@
	tcn_74543:@
	tcn_74545:@@
	tcn_74547:@
	tcn_74549:@@
	tcn_74551:@
	tcn_74553:@@
	tcn_74555:@
	tcn_74557:@@
	tcn_74559:@
	tcn_74561:@@
	tcn_74563:@
	tcn_74565:@@
	tcn_74567:@
	tcn_74569:@@
	tcn_74571:@
	tcn_74573:@@
	tcn_74575:@
	tcn_74577:@@
	tcn_74579:@
dense_74582:@
dense_74584:
identity¢dense/StatefulPartitionedCall¢tcn/StatefulPartitionedCall
tcn/StatefulPartitionedCallStatefulPartitionedCall	tcn_input	tcn_74529	tcn_74531	tcn_74533	tcn_74535	tcn_74537	tcn_74539	tcn_74541	tcn_74543	tcn_74545	tcn_74547	tcn_74549	tcn_74551	tcn_74553	tcn_74555	tcn_74557	tcn_74559	tcn_74561	tcn_74563	tcn_74565	tcn_74567	tcn_74569	tcn_74571	tcn_74573	tcn_74575	tcn_74577	tcn_74579*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_tcn_layer_call_and_return_conditional_losses_73598ÿ
dense/StatefulPartitionedCallStatefulPartitionedCall$tcn/StatefulPartitionedCall:output:0dense_74582dense_74584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73662u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^tcn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
tcn/StatefulPartitionedCalltcn/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
#
_user_specified_name	tcn_input
÷
F
*__inference_SDropout_1_layer_call_fn_76656

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73076v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Åò
±
>__inference_tcn_layer_call_and_return_conditional_losses_76487

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:@b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_1_biasadd_readvariableop_resource:@
identity¢0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Æ
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@û
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_0/SDropout_0/ShapeShape0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_0/SDropout_0/strided_sliceStridedSlice*residual_block_0/SDropout_0/Shape:output:08residual_block_0/SDropout_0/strided_slice/stack:output:0:residual_block_0/SDropout_0/strided_slice/stack_1:output:0:residual_block_0/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_0/SDropout_0/strided_slice_1StridedSlice*residual_block_0/SDropout_0/Shape:output:0:residual_block_0/SDropout_0/strided_slice_1/stack:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_0/conv1D_1/PadPad0residual_block_0/Act_Conv1D_0/Relu:activations:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@Æ
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_0/SDropout_1/ShapeShape0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_0/SDropout_1/strided_sliceStridedSlice*residual_block_0/SDropout_1/Shape:output:08residual_block_0/SDropout_1/strided_slice/stack:output:0:residual_block_0/SDropout_1/strided_slice/stack_1:output:0:residual_block_0/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_0/SDropout_1/strided_slice_1StridedSlice*residual_block_0/SDropout_1/Shape:output:0:residual_block_0/SDropout_1/strided_slice_1/stack:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_0/Act_Conv_Blocks/ReluRelu0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÃ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0Ô
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
Â
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ´
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ä
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_1/SDropout_0/ShapeShape0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_1/SDropout_0/strided_sliceStridedSlice*residual_block_1/SDropout_0/Shape:output:08residual_block_1/SDropout_0/strided_slice/stack:output:0:residual_block_1/SDropout_0/strided_slice/stack_1:output:0:residual_block_1/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_1/SDropout_0/strided_slice_1StridedSlice*residual_block_1/SDropout_0/Shape:output:0:residual_block_1/SDropout_0/strided_slice_1/stack:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_1/conv1D_1/PadPad0residual_block_1/Act_Conv1D_0/Relu:activations:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_1/SDropout_1/ShapeShape0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_1/SDropout_1/strided_sliceStridedSlice*residual_block_1/SDropout_1/Shape:output:08residual_block_1/SDropout_1/strided_slice/stack:output:0:residual_block_1/SDropout_1/strided_slice/stack_1:output:0:residual_block_1/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_1/SDropout_1/strided_slice_1StridedSlice*residual_block_1/SDropout_1/Shape:output:0:residual_block_1/SDropout_1/strided_slice_1/stack:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_1/Act_Conv_Blocks/ReluRelu0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_2/SDropout_0/ShapeShape0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_2/SDropout_0/strided_sliceStridedSlice*residual_block_2/SDropout_0/Shape:output:08residual_block_2/SDropout_0/strided_slice/stack:output:0:residual_block_2/SDropout_0/strided_slice/stack_1:output:0:residual_block_2/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_2/SDropout_0/strided_slice_1StridedSlice*residual_block_2/SDropout_0/Shape:output:0:residual_block_2/SDropout_0/strided_slice_1/stack:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_2/conv1D_1/PadPad0residual_block_2/Act_Conv1D_0/Relu:activations:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_2/SDropout_1/ShapeShape0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_2/SDropout_1/strided_sliceStridedSlice*residual_block_2/SDropout_1/Shape:output:08residual_block_2/SDropout_1/strided_slice/stack:output:0:residual_block_2/SDropout_1/strided_slice/stack_1:output:0:residual_block_2/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_2/SDropout_1/strided_slice_1StridedSlice*residual_block_2/SDropout_1/Shape:output:0:residual_block_2/SDropout_1/strided_slice_1/stack:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_2/Act_Conv_Blocks/ReluRelu0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_3/SDropout_0/ShapeShape0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_3/SDropout_0/strided_sliceStridedSlice*residual_block_3/SDropout_0/Shape:output:08residual_block_3/SDropout_0/strided_slice/stack:output:0:residual_block_3/SDropout_0/strided_slice/stack_1:output:0:residual_block_3/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_3/SDropout_0/strided_slice_1StridedSlice*residual_block_3/SDropout_0/Shape:output:0:residual_block_3/SDropout_0/strided_slice_1/stack:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_3/conv1D_1/PadPad0residual_block_3/Act_Conv1D_0/Relu:activations:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_3/SDropout_1/ShapeShape0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_3/SDropout_1/strided_sliceStridedSlice*residual_block_3/SDropout_1/Shape:output:08residual_block_3/SDropout_1/strided_slice/stack:output:0:residual_block_3/SDropout_1/strided_slice/stack_1:output:0:residual_block_3/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_3/SDropout_1/strided_slice_1StridedSlice*residual_block_3/SDropout_1/Shape:output:0:residual_block_3/SDropout_1/strided_slice_1/stack:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_3/Act_Conv_Blocks/ReluRelu0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ¾
residual_block_4/conv1D_0/PadPad1residual_block_3/Act_Res_Block/Relu:activations:0/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_0/Pad:output:0Dresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_0/Conv1DConv2D4residual_block_4/conv1D_0/Conv1D/ExpandDims:output:06residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_0/BiasAddBiasAdd8residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_0/ReluRelu*residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_4/SDropout_0/ShapeShape0residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_4/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_4/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_4/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_4/SDropout_0/strided_sliceStridedSlice*residual_block_4/SDropout_0/Shape:output:08residual_block_4/SDropout_0/strided_slice/stack:output:0:residual_block_4/SDropout_0/strided_slice/stack_1:output:0:residual_block_4/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_4/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_4/SDropout_0/strided_slice_1StridedSlice*residual_block_4/SDropout_0/Shape:output:0:residual_block_4/SDropout_0/strided_slice_1/stack:output:0<residual_block_4/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_4/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ½
residual_block_4/conv1D_1/PadPad0residual_block_4/Act_Conv1D_0/Relu:activations:0/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_1/Pad:output:0Dresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_1/Conv1DConv2D4residual_block_4/conv1D_1/Conv1D/ExpandDims:output:06residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_1/BiasAddBiasAdd8residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_1/ReluRelu*residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_4/SDropout_1/ShapeShape0residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_4/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_4/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_4/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_4/SDropout_1/strided_sliceStridedSlice*residual_block_4/SDropout_1/Shape:output:08residual_block_4/SDropout_1/strided_slice/stack:output:0:residual_block_4/SDropout_1/strided_slice/stack_1:output:0:residual_block_4/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_4/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_4/SDropout_1/strided_slice_1StridedSlice*residual_block_4/SDropout_1/Shape:output:0:residual_block_4/SDropout_1/strided_slice_1/stack:output:0<residual_block_4/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_4/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_4/Act_Conv_Blocks/ReluRelu0residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_4/Add_Res/addAddV21residual_block_3/Act_Res_Block/Relu:activations:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_4/Act_Res_Block/ReluRelu residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ¾
residual_block_5/conv1D_0/PadPad1residual_block_4/Act_Res_Block/Relu:activations:0/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_0/Pad:output:0Dresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_0/Conv1DConv2D4residual_block_5/conv1D_0/Conv1D/ExpandDims:output:06residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_0/BiasAddBiasAdd8residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_0/ReluRelu*residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_5/SDropout_0/ShapeShape0residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_5/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_5/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_5/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_5/SDropout_0/strided_sliceStridedSlice*residual_block_5/SDropout_0/Shape:output:08residual_block_5/SDropout_0/strided_slice/stack:output:0:residual_block_5/SDropout_0/strided_slice/stack_1:output:0:residual_block_5/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_5/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_5/SDropout_0/strided_slice_1StridedSlice*residual_block_5/SDropout_0/Shape:output:0:residual_block_5/SDropout_0/strided_slice_1/stack:output:0<residual_block_5/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_5/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ½
residual_block_5/conv1D_1/PadPad0residual_block_5/Act_Conv1D_0/Relu:activations:0/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_1/Pad:output:0Dresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_1/Conv1DConv2D4residual_block_5/conv1D_1/Conv1D/ExpandDims:output:06residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_1/BiasAddBiasAdd8residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_1/ReluRelu*residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_5/SDropout_1/ShapeShape0residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_5/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_5/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_5/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_5/SDropout_1/strided_sliceStridedSlice*residual_block_5/SDropout_1/Shape:output:08residual_block_5/SDropout_1/strided_slice/stack:output:0:residual_block_5/SDropout_1/strided_slice/stack_1:output:0:residual_block_5/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_5/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_5/SDropout_1/strided_slice_1StridedSlice*residual_block_5/SDropout_1/Shape:output:0:residual_block_5/SDropout_1/strided_slice_1/stack:output:0<residual_block_5/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_5/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_5/Act_Conv_Blocks/ReluRelu0residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_5/Add_Res/addAddV21residual_block_4/Act_Res_Block/Relu:activations:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_5/Act_Res_Block/ReluRelu residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Á
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¬
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_3AddV2Add_Skip_Connections/add_2:z:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_4AddV2Add_Skip_Connections/add_3:z:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         È
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_4:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76595

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_72998v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76600

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73016v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

*__inference_sequential_layer_call_fn_74778

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_74406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76633

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
² 
E__inference_sequential_layer_call_and_return_conditional_losses_75550

inputs_
Itcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@K
=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:@f
Ptcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@R
Dtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@_
Itcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_4_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_4_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_5_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_5_conv1d_1_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
*tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       
!tcn/residual_block_0/conv1D_0/PadPadinputs3tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
3tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿá
/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_0/Pad:output:0<tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Î
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0w
5tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
$tcn/residual_block_0/conv1D_0/Conv1DConv2D8tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
¼
,tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ®
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Û
%tcn/residual_block_0/conv1D_0/BiasAddBiasAdd5tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_0/Act_Conv1D_0/ReluRelu.tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_0/SDropout_0/ShapeShape4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_0/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_0/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_0/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_0/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_0/SDropout_0/Shape:output:0<tcn/residual_block_0/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_0/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_0/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_0/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_0/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_0/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_0/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_0/SDropout_0/Shape:output:0>tcn/residual_block_0/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_0/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_0/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       É
!tcn/residual_block_0/conv1D_1/PadPad4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:03tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@~
3tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿá
/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_1/Pad:output:0<tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@Î
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_0/conv1D_1/Conv1DConv2D8tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
¼
,tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ®
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Û
%tcn/residual_block_0/conv1D_1/BiasAddBiasAdd5tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_0/Act_Conv1D_1/ReluRelu.tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_0/SDropout_1/ShapeShape4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_0/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_0/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_0/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_0/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_0/SDropout_1/Shape:output:0<tcn/residual_block_0/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_0/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_0/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_0/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_0/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_0/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_0/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_0/SDropout_1/Shape:output:0>tcn/residual_block_0/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_0/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_0/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)tcn/residual_block_0/Act_Conv_Blocks/ReluRelu4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
:tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿË
6tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputsCtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0Ü
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpPtcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0~
<tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
8tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsOtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Etcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
+tcn/residual_block_0/matching_conv1D/Conv1DConv2D?tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Atcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
Ê
3tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze4tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¼
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpDtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ð
,tcn/residual_block_0/matching_conv1D/BiasAddBiasAdd<tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Ctcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_0/Add_Res/addAddV25tcn/residual_block_0/matching_conv1D/BiasAdd:output:07tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_0/Act_Res_Block/ReluRelu$tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ê
!tcn/residual_block_1/conv1D_0/PadPad5tcn/residual_block_0/Act_Res_Block/Relu:activations:03tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@|
2tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¬
Stcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_0/Pad:output:0Htcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_1/conv1D_0/Conv1DConv2D8tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_1/conv1D_0/BiasAddBiasAdd<tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_1/Act_Conv1D_0/ReluRelu.tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_1/SDropout_0/ShapeShape4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_1/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_1/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_1/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_1/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_1/SDropout_0/Shape:output:0<tcn/residual_block_1/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_1/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_1/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_1/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_1/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_1/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_1/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_1/SDropout_0/Shape:output:0>tcn/residual_block_1/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_1/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_1/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       É
!tcn/residual_block_1/conv1D_1/PadPad4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:03tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@|
2tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¬
Stcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_1/Pad:output:0Htcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_1/conv1D_1/Conv1DConv2D8tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_1/conv1D_1/BiasAddBiasAdd<tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_1/Act_Conv1D_1/ReluRelu.tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_1/SDropout_1/ShapeShape4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_1/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_1/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_1/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_1/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_1/SDropout_1/Shape:output:0<tcn/residual_block_1/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_1/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_1/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_1/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_1/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_1/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_1/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_1/SDropout_1/Shape:output:0>tcn/residual_block_1/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_1/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_1/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)tcn/residual_block_1/Act_Conv_Blocks/ReluRelu4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_1/Add_Res/addAddV25tcn/residual_block_0/Act_Res_Block/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_1/Act_Res_Block/ReluRelu$tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ê
!tcn/residual_block_2/conv1D_0/PadPad5tcn/residual_block_1/Act_Res_Block/Relu:activations:03tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@|
2tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¬
Stcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_0/Pad:output:0Htcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_2/conv1D_0/Conv1DConv2D8tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_2/conv1D_0/BiasAddBiasAdd<tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_2/Act_Conv1D_0/ReluRelu.tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_2/SDropout_0/ShapeShape4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_2/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_2/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_2/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_2/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_2/SDropout_0/Shape:output:0<tcn/residual_block_2/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_2/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_2/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_2/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_2/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_2/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_2/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_2/SDropout_0/Shape:output:0>tcn/residual_block_2/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_2/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_2/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       É
!tcn/residual_block_2/conv1D_1/PadPad4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:03tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@|
2tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¬
Stcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_1/Pad:output:0Htcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_2/conv1D_1/Conv1DConv2D8tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_2/conv1D_1/BiasAddBiasAdd<tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_2/Act_Conv1D_1/ReluRelu.tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_2/SDropout_1/ShapeShape4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_2/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_2/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_2/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_2/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_2/SDropout_1/Shape:output:0<tcn/residual_block_2/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_2/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_2/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_2/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_2/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_2/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_2/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_2/SDropout_1/Shape:output:0>tcn/residual_block_2/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_2/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_2/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)tcn/residual_block_2/Act_Conv_Blocks/ReluRelu4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_2/Add_Res/addAddV25tcn/residual_block_1/Act_Res_Block/Relu:activations:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_2/Act_Res_Block/ReluRelu$tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ê
!tcn/residual_block_3/conv1D_0/PadPad5tcn/residual_block_2/Act_Res_Block/Relu:activations:03tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@|
2tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¬
Stcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_0/Pad:output:0Htcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_3/conv1D_0/Conv1DConv2D8tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_3/conv1D_0/BiasAddBiasAdd<tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_3/Act_Conv1D_0/ReluRelu.tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_3/SDropout_0/ShapeShape4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_3/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_3/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_3/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_3/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_3/SDropout_0/Shape:output:0<tcn/residual_block_3/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_3/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_3/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_3/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_3/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_3/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_3/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_3/SDropout_0/Shape:output:0>tcn/residual_block_3/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_3/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_3/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       É
!tcn/residual_block_3/conv1D_1/PadPad4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:03tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@|
2tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¬
Stcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_1/Pad:output:0Htcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_3/conv1D_1/Conv1DConv2D8tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_3/conv1D_1/BiasAddBiasAdd<tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_3/Act_Conv1D_1/ReluRelu.tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_3/SDropout_1/ShapeShape4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_3/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_3/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_3/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_3/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_3/SDropout_1/Shape:output:0<tcn/residual_block_3/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_3/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_3/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_3/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_3/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_3/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_3/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_3/SDropout_1/Shape:output:0>tcn/residual_block_3/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_3/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_3/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)tcn/residual_block_3/Act_Conv_Blocks/ReluRelu4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_3/Add_Res/addAddV25tcn/residual_block_2/Act_Res_Block/Relu:activations:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_3/Act_Res_Block/ReluRelu$tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        Ê
!tcn/residual_block_4/conv1D_0/PadPad5tcn/residual_block_3/Act_Res_Block/Relu:activations:03tcn/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@|
2tcn/residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¬
Stcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_4/conv1D_0/Pad:output:0Htcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_4/conv1D_0/Conv1DConv2D8tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_4/conv1D_0/BiasAddBiasAdd<tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_4/Act_Conv1D_0/ReluRelu.tcn/residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_4/SDropout_0/ShapeShape4tcn/residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_4/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_4/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_4/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_4/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_4/SDropout_0/Shape:output:0<tcn/residual_block_4/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_4/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_4/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_4/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_4/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_4/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_4/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_4/SDropout_0/Shape:output:0>tcn/residual_block_4/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_4/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_4/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*tcn/residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        É
!tcn/residual_block_4/conv1D_1/PadPad4tcn/residual_block_4/Act_Conv1D_0/Relu:activations:03tcn/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@|
2tcn/residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¬
Stcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_4/conv1D_1/Pad:output:0Htcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_4/conv1D_1/Conv1DConv2D8tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_4/conv1D_1/BiasAddBiasAdd<tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_4/Act_Conv1D_1/ReluRelu.tcn/residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_4/SDropout_1/ShapeShape4tcn/residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_4/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_4/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_4/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_4/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_4/SDropout_1/Shape:output:0<tcn/residual_block_4/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_4/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_4/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_4/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_4/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_4/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_4/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_4/SDropout_1/Shape:output:0>tcn/residual_block_4/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_4/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_4/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)tcn/residual_block_4/Act_Conv_Blocks/ReluRelu4tcn/residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_4/Add_Res/addAddV25tcn/residual_block_3/Act_Res_Block/Relu:activations:07tcn/residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_4/Act_Res_Block/ReluRelu$tcn/residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               Ê
!tcn/residual_block_5/conv1D_0/PadPad5tcn/residual_block_4/Act_Res_Block/Relu:activations:03tcn/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@|
2tcn/residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Qtcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¬
Stcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¤
Ktcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
?tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
<tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¸
3tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_5/conv1D_0/Pad:output:0Htcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_5/conv1D_0/Conv1DConv2D8tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
9tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       À
3tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_5/conv1D_0/BiasAddBiasAdd<tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_5/Act_Conv1D_0/ReluRelu.tcn/residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_5/SDropout_0/ShapeShape4tcn/residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_5/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_5/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_5/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_5/SDropout_0/strided_sliceStridedSlice.tcn/residual_block_5/SDropout_0/Shape:output:0<tcn/residual_block_5/SDropout_0/strided_slice/stack:output:0>tcn/residual_block_5/SDropout_0/strided_slice/stack_1:output:0>tcn/residual_block_5/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_5/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_5/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_5/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_5/SDropout_0/strided_slice_1StridedSlice.tcn/residual_block_5/SDropout_0/Shape:output:0>tcn/residual_block_5/SDropout_0/strided_slice_1/stack:output:0@tcn/residual_block_5/SDropout_0/strided_slice_1/stack_1:output:0@tcn/residual_block_5/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*tcn/residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               É
!tcn/residual_block_5/conv1D_1/PadPad4tcn/residual_block_5/Act_Conv1D_0/Relu:activations:03tcn/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@|
2tcn/residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Qtcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¬
Stcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¤
Ktcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
?tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
<tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¸
3tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_5/conv1D_1/Pad:output:0Htcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_5/conv1D_1/Conv1DConv2D8tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
9tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       À
3tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_5/conv1D_1/BiasAddBiasAdd<tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_5/Act_Conv1D_1/ReluRelu.tcn/residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
%tcn/residual_block_5/SDropout_1/ShapeShape4tcn/residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:}
3tcn/residual_block_5/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tcn/residual_block_5/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tcn/residual_block_5/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-tcn/residual_block_5/SDropout_1/strided_sliceStridedSlice.tcn/residual_block_5/SDropout_1/Shape:output:0<tcn/residual_block_5/SDropout_1/strided_slice/stack:output:0>tcn/residual_block_5/SDropout_1/strided_slice/stack_1:output:0>tcn/residual_block_5/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5tcn/residual_block_5/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_5/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7tcn/residual_block_5/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/tcn/residual_block_5/SDropout_1/strided_slice_1StridedSlice.tcn/residual_block_5/SDropout_1/Shape:output:0>tcn/residual_block_5/SDropout_1/strided_slice_1/stack:output:0@tcn/residual_block_5/SDropout_1/strided_slice_1/stack_1:output:0@tcn/residual_block_5/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)tcn/residual_block_5/Act_Conv_Blocks/ReluRelu4tcn/residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_5/Add_Res/addAddV25tcn/residual_block_4/Act_Res_Block/Relu:activations:07tcn/residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_5/Act_Res_Block/ReluRelu$tcn/residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Í
tcn/Add_Skip_Connections/addAddV27tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¸
tcn/Add_Skip_Connections/add_1AddV2 tcn/Add_Skip_Connections/add:z:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@º
tcn/Add_Skip_Connections/add_2AddV2"tcn/Add_Skip_Connections/add_1:z:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@º
tcn/Add_Skip_Connections/add_3AddV2"tcn/Add_Skip_Connections/add_2:z:07tcn/residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@º
tcn/Add_Skip_Connections/add_4AddV2"tcn/Add_Skip_Connections/add_3:z:07tcn/residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@y
$tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    {
&tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            {
&tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ü
tcn/Slice_Output/strided_sliceStridedSlice"tcn/Add_Skip_Connections/add_4:z:0-tcn/Slice_Output/strided_slice/stack:output:0/tcn/Slice_Output/strided_slice/stack_1:output:0/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_mask
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul'tcn/Slice_Output/strided_slice:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp5^tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<^tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpH^tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2l
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2z
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpGtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Åò
±
>__inference_tcn_layer_call_and_return_conditional_losses_74226

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource:@b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_4_conv1d_1_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_0_biasadd_readvariableop_resource:@[
Eresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@G
9residual_block_5_conv1d_1_biasadd_readvariableop_resource:@
identity¢0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Æ
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@û
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_0/SDropout_0/ShapeShape0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_0/SDropout_0/strided_sliceStridedSlice*residual_block_0/SDropout_0/Shape:output:08residual_block_0/SDropout_0/strided_slice/stack:output:0:residual_block_0/SDropout_0/strided_slice/stack_1:output:0:residual_block_0/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_0/SDropout_0/strided_slice_1StridedSlice*residual_block_0/SDropout_0/Shape:output:0:residual_block_0/SDropout_0/strided_slice_1/stack:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_0/conv1D_1/PadPad0residual_block_0/Act_Conv1D_0/Relu:activations:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@Æ
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
´
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_0/SDropout_1/ShapeShape0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_0/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_0/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_0/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_0/SDropout_1/strided_sliceStridedSlice*residual_block_0/SDropout_1/Shape:output:08residual_block_0/SDropout_1/strided_slice/stack:output:0:residual_block_0/SDropout_1/strided_slice/stack_1:output:0:residual_block_0/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_0/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_0/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_0/SDropout_1/strided_slice_1StridedSlice*residual_block_0/SDropout_1/Shape:output:0:residual_block_0/SDropout_1/strided_slice_1/stack:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_0/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_0/Act_Conv_Blocks/ReluRelu0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÃ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0Ô
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
Â
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ´
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ä
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_1/SDropout_0/ShapeShape0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_1/SDropout_0/strided_sliceStridedSlice*residual_block_1/SDropout_0/Shape:output:08residual_block_1/SDropout_0/strided_slice/stack:output:0:residual_block_1/SDropout_0/strided_slice/stack_1:output:0:residual_block_1/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_1/SDropout_0/strided_slice_1StridedSlice*residual_block_1/SDropout_0/Shape:output:0:residual_block_1/SDropout_0/strided_slice_1/stack:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_1/conv1D_1/PadPad0residual_block_1/Act_Conv1D_0/Relu:activations:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¨
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_1/SDropout_1/ShapeShape0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_1/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_1/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_1/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_1/SDropout_1/strided_sliceStridedSlice*residual_block_1/SDropout_1/Shape:output:08residual_block_1/SDropout_1/strided_slice/stack:output:0:residual_block_1/SDropout_1/strided_slice/stack_1:output:0:residual_block_1/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_1/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_1/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_1/SDropout_1/strided_slice_1StridedSlice*residual_block_1/SDropout_1/Shape:output:0:residual_block_1/SDropout_1/strided_slice_1/stack:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_1/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_1/Act_Conv_Blocks/ReluRelu0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_2/SDropout_0/ShapeShape0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_2/SDropout_0/strided_sliceStridedSlice*residual_block_2/SDropout_0/Shape:output:08residual_block_2/SDropout_0/strided_slice/stack:output:0:residual_block_2/SDropout_0/strided_slice/stack_1:output:0:residual_block_2/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_2/SDropout_0/strided_slice_1StridedSlice*residual_block_2/SDropout_0/Shape:output:0:residual_block_2/SDropout_0/strided_slice_1/stack:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_2/conv1D_1/PadPad0residual_block_2/Act_Conv1D_0/Relu:activations:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¨
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_2/SDropout_1/ShapeShape0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_2/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_2/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_2/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_2/SDropout_1/strided_sliceStridedSlice*residual_block_2/SDropout_1/Shape:output:08residual_block_2/SDropout_1/strided_slice/stack:output:0:residual_block_2/SDropout_1/strided_slice/stack_1:output:0:residual_block_2/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_2/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_2/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_2/SDropout_1/strided_slice_1StridedSlice*residual_block_2/SDropout_1/Shape:output:0:residual_block_2/SDropout_1/strided_slice_1/stack:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_2/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_2/Act_Conv_Blocks/ReluRelu0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ¾
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_3/SDropout_0/ShapeShape0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_3/SDropout_0/strided_sliceStridedSlice*residual_block_3/SDropout_0/Shape:output:08residual_block_3/SDropout_0/strided_slice/stack:output:0:residual_block_3/SDropout_0/strided_slice/stack_1:output:0:residual_block_3/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_3/SDropout_0/strided_slice_1StridedSlice*residual_block_3/SDropout_0/Shape:output:0:residual_block_3/SDropout_0/strided_slice_1/stack:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ½
residual_block_3/conv1D_1/PadPad0residual_block_3/Act_Conv1D_0/Relu:activations:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¨
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_3/SDropout_1/ShapeShape0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_3/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_3/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_3/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_3/SDropout_1/strided_sliceStridedSlice*residual_block_3/SDropout_1/Shape:output:08residual_block_3/SDropout_1/strided_slice/stack:output:0:residual_block_3/SDropout_1/strided_slice/stack_1:output:0:residual_block_3/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_3/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_3/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_3/SDropout_1/strided_slice_1StridedSlice*residual_block_3/SDropout_1/Shape:output:0:residual_block_3/SDropout_1/strided_slice_1/stack:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_3/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_3/Act_Conv_Blocks/ReluRelu0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ¾
residual_block_4/conv1D_0/PadPad1residual_block_3/Act_Res_Block/Relu:activations:0/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_0/Pad:output:0Dresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_0/Conv1DConv2D4residual_block_4/conv1D_0/Conv1D/ExpandDims:output:06residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_0/BiasAddBiasAdd8residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_0/ReluRelu*residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_4/SDropout_0/ShapeShape0residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_4/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_4/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_4/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_4/SDropout_0/strided_sliceStridedSlice*residual_block_4/SDropout_0/Shape:output:08residual_block_4/SDropout_0/strided_slice/stack:output:0:residual_block_4/SDropout_0/strided_slice/stack_1:output:0:residual_block_4/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_4/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_4/SDropout_0/strided_slice_1StridedSlice*residual_block_4/SDropout_0/Shape:output:0:residual_block_4/SDropout_0/strided_slice_1/stack:output:0<residual_block_4/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_4/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        ½
residual_block_4/conv1D_1/PadPad0residual_block_4/Act_Conv1D_0/Relu:activations:0/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@x
.residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Mresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¨
Oresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"         
Gresidual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
;residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¨
/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_4/conv1D_1/Pad:output:0Dresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_4/conv1D_1/Conv1DConv2D4residual_block_4/conv1D_1/Conv1D/ExpandDims:output:06residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
5residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        °
/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_4/conv1D_1/BiasAddBiasAdd8residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_4/Act_Conv1D_1/ReluRelu*residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_4/SDropout_1/ShapeShape0residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_4/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_4/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_4/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_4/SDropout_1/strided_sliceStridedSlice*residual_block_4/SDropout_1/Shape:output:08residual_block_4/SDropout_1/strided_slice/stack:output:0:residual_block_4/SDropout_1/strided_slice/stack_1:output:0:residual_block_4/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_4/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_4/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_4/SDropout_1/strided_slice_1StridedSlice*residual_block_4/SDropout_1/Shape:output:0:residual_block_4/SDropout_1/strided_slice_1/stack:output:0<residual_block_4/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_4/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_4/Act_Conv_Blocks/ReluRelu0residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_4/Add_Res/addAddV21residual_block_3/Act_Res_Block/Relu:activations:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_4/Act_Res_Block/ReluRelu residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ¾
residual_block_5/conv1D_0/PadPad1residual_block_4/Act_Res_Block/Relu:activations:0/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_0/Pad:output:0Dresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_0/Conv1DConv2D4residual_block_5/conv1D_0/Conv1D/ExpandDims:output:06residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_0/BiasAddBiasAdd8residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_0/ReluRelu*residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_5/SDropout_0/ShapeShape0residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_5/SDropout_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_5/SDropout_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_5/SDropout_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_5/SDropout_0/strided_sliceStridedSlice*residual_block_5/SDropout_0/Shape:output:08residual_block_5/SDropout_0/strided_slice/stack:output:0:residual_block_5/SDropout_0/strided_slice/stack_1:output:0:residual_block_5/SDropout_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_5/SDropout_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_5/SDropout_0/strided_slice_1StridedSlice*residual_block_5/SDropout_0/Shape:output:0:residual_block_5/SDropout_0/strided_slice_1/stack:output:0<residual_block_5/SDropout_0/strided_slice_1/stack_1:output:0<residual_block_5/SDropout_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               ½
residual_block_5/conv1D_1/PadPad0residual_block_5/Act_Conv1D_0/Relu:activations:0/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@x
.residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Mresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¨
Oresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        
Gresidual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
;residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¨
/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_5/conv1D_1/Pad:output:0Dresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
+residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Æ
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0s
1residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@û
 residual_block_5/conv1D_1/Conv1DConv2D4residual_block_5/conv1D_1/Conv1D/ExpandDims:output:06residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
´
(residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
;residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
5residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       °
/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¦
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
!residual_block_5/conv1D_1/BiasAddBiasAdd8residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"residual_block_5/Act_Conv1D_1/ReluRelu*residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
!residual_block_5/SDropout_1/ShapeShape0residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*
_output_shapes
:y
/residual_block_5/SDropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1residual_block_5/SDropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1residual_block_5/SDropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)residual_block_5/SDropout_1/strided_sliceStridedSlice*residual_block_5/SDropout_1/Shape:output:08residual_block_5/SDropout_1/strided_slice/stack:output:0:residual_block_5/SDropout_1/strided_slice/stack_1:output:0:residual_block_5/SDropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1residual_block_5/SDropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3residual_block_5/SDropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+residual_block_5/SDropout_1/strided_slice_1StridedSlice*residual_block_5/SDropout_1/Shape:output:0:residual_block_5/SDropout_1/strided_slice_1/stack:output:0<residual_block_5/SDropout_1/strided_slice_1/stack_1:output:0<residual_block_5/SDropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%residual_block_5/Act_Conv_Blocks/ReluRelu0residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ã
residual_block_5/Add_Res/addAddV21residual_block_4/Act_Res_Block/Relu:activations:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
#residual_block_5/Act_Res_Block/ReluRelu residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Á
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¬
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_3AddV2Add_Skip_Connections/add_2:z:03residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
Add_Skip_Connections/add_4AddV2Add_Skip_Connections/add_3:z:03residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         È
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_4:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_4/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_5/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp0residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp0residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp0residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp0residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_0_layer_call_fn_76791

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73208v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
² 
E__inference_sequential_layer_call_and_return_conditional_losses_75116

inputs_
Itcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@K
=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource:@f
Ptcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource:@R
Dtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource:@_
Itcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_4_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_4_conv1d_1_biasadd_readvariableop_resource:@_
Itcn_residual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_5_conv1d_0_biasadd_readvariableop_resource:@_
Itcn_residual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@K
=tcn_residual_block_5_conv1d_1_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp¢Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp¢@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp¢4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp¢@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp
*tcn/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       
!tcn/residual_block_0/conv1D_0/PadPadinputs3tcn/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
3tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿá
/tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_0/Pad:output:0<tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Î
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0w
5tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
$tcn/residual_block_0/conv1D_0/Conv1DConv2D8tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
¼
,tcn/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ®
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Û
%tcn/residual_block_0/conv1D_0/BiasAddBiasAdd5tcn/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_0/Act_Conv1D_0/ReluRelu.tcn/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_0/SDropout_0/IdentityIdentity4tcn/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Æ
!tcn/residual_block_0/conv1D_1/PadPad1tcn/residual_block_0/SDropout_0/Identity:output:03tcn/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@~
3tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿá
/tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims*tcn/residual_block_0/conv1D_1/Pad:output:0<tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@Î
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_0/conv1D_1/Conv1DConv2D8tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
¼
,tcn/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_0/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ®
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Û
%tcn/residual_block_0/conv1D_1/BiasAddBiasAdd5tcn/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0<tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_0/Act_Conv1D_1/ReluRelu.tcn/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_0/SDropout_1/IdentityIdentity4tcn/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
)tcn/residual_block_0/Act_Conv_Blocks/ReluRelu1tcn/residual_block_0/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
:tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿË
6tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputsCtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0Ü
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpPtcn_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0~
<tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
8tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsOtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Etcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@
+tcn/residual_block_0/matching_conv1D/Conv1DConv2D?tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Atcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
Ê
3tcn/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze4tcn/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ¼
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpDtcn_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ð
,tcn/residual_block_0/matching_conv1D/BiasAddBiasAdd<tcn/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Ctcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_0/Add_Res/addAddV25tcn/residual_block_0/matching_conv1D/BiasAdd:output:07tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_0/Act_Res_Block/ReluRelu$tcn/residual_block_0/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ê
!tcn/residual_block_1/conv1D_0/PadPad5tcn/residual_block_0/Act_Res_Block/Relu:activations:03tcn/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@|
2tcn/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¬
Stcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_0/Pad:output:0Htcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_1/conv1D_0/Conv1DConv2D8tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_1/conv1D_0/BiasAddBiasAdd<tcn/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_1/Act_Conv1D_0/ReluRelu.tcn/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_1/SDropout_0/IdentityIdentity4tcn/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Æ
!tcn/residual_block_1/conv1D_1/PadPad1tcn/residual_block_1/SDropout_0/Identity:output:03tcn/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@|
2tcn/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:4¬
Stcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_1/conv1D_1/Pad:output:0Htcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_1/conv1D_1/Conv1DConv2D8tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_1/conv1D_1/BiasAddBiasAdd<tcn/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_1/Act_Conv1D_1/ReluRelu.tcn/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_1/SDropout_1/IdentityIdentity4tcn/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
)tcn/residual_block_1/Act_Conv_Blocks/ReluRelu1tcn/residual_block_1/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_1/Add_Res/addAddV25tcn/residual_block_0/Act_Res_Block/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_1/Act_Res_Block/ReluRelu$tcn/residual_block_1/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ê
!tcn/residual_block_2/conv1D_0/PadPad5tcn/residual_block_1/Act_Res_Block/Relu:activations:03tcn/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@|
2tcn/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¬
Stcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_0/Pad:output:0Htcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_2/conv1D_0/Conv1DConv2D8tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_2/conv1D_0/BiasAddBiasAdd<tcn/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_2/Act_Conv1D_0/ReluRelu.tcn/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_2/SDropout_0/IdentityIdentity4tcn/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Æ
!tcn/residual_block_2/conv1D_1/PadPad1tcn/residual_block_2/SDropout_0/Identity:output:03tcn/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8@|
2tcn/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:8¬
Stcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_2/conv1D_1/Pad:output:0Htcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_2/conv1D_1/Conv1DConv2D8tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_2/conv1D_1/BiasAddBiasAdd<tcn/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_2/Act_Conv1D_1/ReluRelu.tcn/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_2/SDropout_1/IdentityIdentity4tcn/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
)tcn/residual_block_2/Act_Conv_Blocks/ReluRelu1tcn/residual_block_2/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_2/Add_Res/addAddV25tcn/residual_block_1/Act_Res_Block/Relu:activations:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_2/Act_Res_Block/ReluRelu$tcn/residual_block_2/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ê
!tcn/residual_block_3/conv1D_0/PadPad5tcn/residual_block_2/Act_Res_Block/Relu:activations:03tcn/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@|
2tcn/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¬
Stcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_0/Pad:output:0Htcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_3/conv1D_0/Conv1DConv2D8tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_3/conv1D_0/BiasAddBiasAdd<tcn/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_3/Act_Conv1D_0/ReluRelu.tcn/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_3/SDropout_0/IdentityIdentity4tcn/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Æ
!tcn/residual_block_3/conv1D_1/PadPad1tcn/residual_block_3/SDropout_0/Identity:output:03tcn/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@|
2tcn/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:@¬
Stcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_3/conv1D_1/Pad:output:0Htcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_3/conv1D_1/Conv1DConv2D8tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_3/conv1D_1/BiasAddBiasAdd<tcn/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_3/Act_Conv1D_1/ReluRelu.tcn/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_3/SDropout_1/IdentityIdentity4tcn/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
)tcn/residual_block_3/Act_Conv_Blocks/ReluRelu1tcn/residual_block_3/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_3/Add_Res/addAddV25tcn/residual_block_2/Act_Res_Block/Relu:activations:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_3/Act_Res_Block/ReluRelu$tcn/residual_block_3/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_4/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        Ê
!tcn/residual_block_4/conv1D_0/PadPad5tcn/residual_block_3/Act_Res_Block/Relu:activations:03tcn/residual_block_4/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@|
2tcn/residual_block_4/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¬
Stcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_4/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_4/conv1D_0/Pad:output:0Htcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_4/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_4_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_4/conv1D_0/Conv1DConv2D8tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_4/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_4/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_4/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_4_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_4/conv1D_0/BiasAddBiasAdd<tcn/residual_block_4/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_4/Act_Conv1D_0/ReluRelu.tcn/residual_block_4/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_4/SDropout_0/IdentityIdentity4tcn/residual_block_4/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_4/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                        Æ
!tcn/residual_block_4/conv1D_1/PadPad1tcn/residual_block_4/SDropout_0/Identity:output:03tcn/residual_block_4/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP@|
2tcn/residual_block_4/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:
Qtcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:P¬
Stcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¤
Ktcn/residual_block_4/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        
?tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
<tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¸
3tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_4/conv1D_1/Pad:output:0Htcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_4/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_4_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_4/conv1D_1/Conv1DConv2D8tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_4/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_4/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:
9tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        À
3tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_4/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_4_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_4/conv1D_1/BiasAddBiasAdd<tcn/residual_block_4/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_4/Act_Conv1D_1/ReluRelu.tcn/residual_block_4/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_4/SDropout_1/IdentityIdentity4tcn/residual_block_4/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
)tcn/residual_block_4/Act_Conv_Blocks/ReluRelu1tcn/residual_block_4/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_4/Add_Res/addAddV25tcn/residual_block_3/Act_Res_Block/Relu:activations:07tcn/residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_4/Act_Res_Block/ReluRelu$tcn/residual_block_4/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_5/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               Ê
!tcn/residual_block_5/conv1D_0/PadPad5tcn/residual_block_4/Act_Res_Block/Relu:activations:03tcn/residual_block_5/conv1D_0/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@|
2tcn/residual_block_5/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Qtcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¬
Stcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¤
Ktcn/residual_block_5/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
?tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
<tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¸
3tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_5/conv1D_0/Pad:output:0Htcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_5/conv1D_0/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_5_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_5/conv1D_0/Conv1DConv2D8tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims:output:0:tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_5/conv1D_0/Conv1D/SqueezeSqueeze-tcn/residual_block_5/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
9tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       À
3tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_5/conv1D_0/Conv1D/Squeeze:output:0Htcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_5_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_5/conv1D_0/BiasAddBiasAdd<tcn/residual_block_5/conv1D_0/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_5/Act_Conv1D_0/ReluRelu.tcn/residual_block_5/conv1D_0/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_5/SDropout_0/IdentityIdentity4tcn/residual_block_5/Act_Conv1D_0/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
*tcn/residual_block_5/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        @               Æ
!tcn/residual_block_5/conv1D_1/PadPad1tcn/residual_block_5/SDropout_0/Identity:output:03tcn/residual_block_5/conv1D_1/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp@|
2tcn/residual_block_5/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB: 
Qtcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:p¬
Stcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        §
Ntcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¤
Ktcn/residual_block_5/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       
?tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
<tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¸
3tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND*tcn/residual_block_5/conv1D_1/Pad:output:0Htcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Etcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
3tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿó
/tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims
ExpandDims<tcn/residual_block_5/conv1D_1/Conv1D/SpaceToBatchND:output:0<tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpItcn_residual_block_5_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0w
5tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsHtcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0>tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
$tcn/residual_block_5/conv1D_1/Conv1DConv2D8tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims:output:0:tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¼
,tcn/residual_block_5/conv1D_1/Conv1D/SqueezeSqueeze-tcn/residual_block_5/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
?tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB: 
9tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       À
3tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND5tcn/residual_block_5/conv1D_1/Conv1D/Squeeze:output:0Htcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Btcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@®
4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp=tcn_residual_block_5_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0â
%tcn/residual_block_5/conv1D_1/BiasAddBiasAdd<tcn/residual_block_5/conv1D_1/Conv1D/BatchToSpaceND:output:0<tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
&tcn/residual_block_5/Act_Conv1D_1/ReluRelu.tcn/residual_block_5/conv1D_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@ 
(tcn/residual_block_5/SDropout_1/IdentityIdentity4tcn/residual_block_5/Act_Conv1D_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
)tcn/residual_block_5/Act_Conv_Blocks/ReluRelu1tcn/residual_block_5/SDropout_1/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Ï
 tcn/residual_block_5/Add_Res/addAddV25tcn/residual_block_4/Act_Res_Block/Relu:activations:07tcn/residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
'tcn/residual_block_5/Act_Res_Block/ReluRelu$tcn/residual_block_5/Add_Res/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@Í
tcn/Add_Skip_Connections/addAddV27tcn/residual_block_0/Act_Conv_Blocks/Relu:activations:07tcn/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@¸
tcn/Add_Skip_Connections/add_1AddV2 tcn/Add_Skip_Connections/add:z:07tcn/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@º
tcn/Add_Skip_Connections/add_2AddV2"tcn/Add_Skip_Connections/add_1:z:07tcn/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@º
tcn/Add_Skip_Connections/add_3AddV2"tcn/Add_Skip_Connections/add_2:z:07tcn/residual_block_4/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@º
tcn/Add_Skip_Connections/add_4AddV2"tcn/Add_Skip_Connections/add_3:z:07tcn/residual_block_5/Act_Conv_Blocks/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@y
$tcn/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    {
&tcn/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            {
&tcn/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ü
tcn/Slice_Output/strided_sliceStridedSlice"tcn/Add_Skip_Connections/add_4:z:0-tcn/Slice_Output/strided_slice/stack:output:0/tcn/Slice_Output/strided_slice/stack_1:output:0/tcn/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

begin_mask*
end_mask*
shrink_axis_mask
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul'tcn/Slice_Output/strided_slice:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp5^tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<^tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpH^tcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOpA^tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp5^tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOpA^tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2l
4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2z
;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp;tcn/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2
Gtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpGtcn/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_4/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_4/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_4/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_4/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp4tcn/residual_block_5/conv1D_0/BiasAdd/ReadVariableOp2
@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_5/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2l
4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp4tcn/residual_block_5/conv1D_1/BiasAdd/ReadVariableOp2
@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp@tcn/residual_block_5/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
°
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73238

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76824

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73256v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76819

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73238v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76758

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76768

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73196v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73076

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73148

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

*__inference_sequential_layer_call_fn_73728
	tcn_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@@

unknown_20:@ 

unknown_21:@@

unknown_22:@ 

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCall	tcn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_73669o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ0: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
#
_user_specified_name	tcn_input
÷
F
*__inference_SDropout_0_layer_call_fn_76516

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_72926v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
F
*__inference_SDropout_1_layer_call_fn_76763

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_73178v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76534

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
a
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76562

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_73028

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*°
serving_default
C
	tcn_input6
serving_default_tcn_input:0ÿÿÿÿÿÿÿÿÿ09
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:£
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
ú
skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
residual_block_3
residual_block_4
residual_block_5
slicer_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer

$iter

%beta_1

&beta_2
	'decay
(learning_ratemÖm×)mØ*mÙ+mÚ,mÛ-mÜ.mÝ/mÞ0mß1mà2má3mâ4mã5mä6må7mæ8mç9mè:mé;mê<më=mì>mí?mî@mïAmðBmñvòvó)vô*võ+vö,v÷-vø.vù/vú0vû1vü2vý3vþ4vÿ5v6v7v8v9v:v;v<v=v>v?v@vAvBv"
	optimizer
ö
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
26
27"
trackable_list_wrapper
ö
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
26
27"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_sequential_layer_call_fn_73728
*__inference_sequential_layer_call_fn_74717
*__inference_sequential_layer_call_fn_74778
*__inference_sequential_layer_call_fn_74526À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_75116
E__inference_sequential_layer_call_and_return_conditional_losses_75550
E__inference_sequential_layer_call_and_return_conditional_losses_74588
E__inference_sequential_layer_call_and_return_conditional_losses_74650À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
 __inference__wrapped_model_72899	tcn_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Hserving_default"
signature_map
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
ú

Ilayers
Jshape_match_conv
Kfinal_activation
Lconv1D_0
MAct_Conv1D_0
N
SDropout_0
Oconv1D_1
PAct_Conv1D_1
Q
SDropout_1
RAct_Conv_Blocks
Jmatching_conv1D
KAct_Res_Block
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
ü

Ylayers
Zshape_match_conv
[final_activation
\conv1D_0
]Act_Conv1D_0
^
SDropout_0
_conv1D_1
`Act_Conv1D_1
a
SDropout_1
bAct_Conv_Blocks
Zmatching_identity
[Act_Res_Block
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
ü

ilayers
jshape_match_conv
kfinal_activation
lconv1D_0
mAct_Conv1D_0
n
SDropout_0
oconv1D_1
pAct_Conv1D_1
q
SDropout_1
rAct_Conv_Blocks
jmatching_identity
kAct_Res_Block
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer


ylayers
zshape_match_conv
{final_activation
|conv1D_0
}Act_Conv1D_0
~
SDropout_0
conv1D_1
Act_Conv1D_1

SDropout_1
Act_Conv_Blocks
zmatching_identity
{Act_Res_Block
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

layers
shape_match_conv
final_activation
conv1D_0
Act_Conv1D_0

SDropout_0
conv1D_1
Act_Conv1D_1

SDropout_1
Act_Conv_Blocks
matching_identity
Act_Res_Block
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

layers
shape_match_conv
final_activation
conv1D_0
Act_Conv1D_0

SDropout_0
conv1D_1
 Act_Conv1D_1
¡
SDropout_1
¢Act_Conv_Blocks
matching_identity
Act_Res_Block
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
«
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
æ
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25"
trackable_list_wrapper
æ
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
#__inference_tcn_layer_call_fn_75670
#__inference_tcn_layer_call_fn_75727º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
>__inference_tcn_layer_call_and_return_conditional_losses_76059
>__inference_tcn_layer_call_and_return_conditional_losses_76487º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:@2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_dense_layer_call_fn_76496¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_76506¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
::8@2$tcn/residual_block_0/conv1D_0/kernel
0:.@2"tcn/residual_block_0/conv1D_0/bias
::8@@2$tcn/residual_block_0/conv1D_1/kernel
0:.@2"tcn/residual_block_0/conv1D_1/bias
A:?@2+tcn/residual_block_0/matching_conv1D/kernel
7:5@2)tcn/residual_block_0/matching_conv1D/bias
::8@@2$tcn/residual_block_1/conv1D_0/kernel
0:.@2"tcn/residual_block_1/conv1D_0/bias
::8@@2$tcn/residual_block_1/conv1D_1/kernel
0:.@2"tcn/residual_block_1/conv1D_1/bias
::8@@2$tcn/residual_block_2/conv1D_0/kernel
0:.@2"tcn/residual_block_2/conv1D_0/bias
::8@@2$tcn/residual_block_2/conv1D_1/kernel
0:.@2"tcn/residual_block_2/conv1D_1/bias
::8@@2$tcn/residual_block_3/conv1D_0/kernel
0:.@2"tcn/residual_block_3/conv1D_0/bias
::8@@2$tcn/residual_block_3/conv1D_1/kernel
0:.@2"tcn/residual_block_3/conv1D_1/bias
::8@@2$tcn/residual_block_4/conv1D_0/kernel
0:.@2"tcn/residual_block_4/conv1D_0/bias
::8@@2$tcn/residual_block_4/conv1D_1/kernel
0:.@2"tcn/residual_block_4/conv1D_1/bias
::8@@2$tcn/residual_block_5/conv1D_0/kernel
0:.@2"tcn/residual_block_5/conv1D_0/bias
::8@@2$tcn/residual_block_5/conv1D_1/kernel
0:.@2"tcn/residual_block_5/conv1D_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
H
¹0
º1
»2
¼3
½4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÌBÉ
#__inference_signature_wrapper_75613	tcn_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Q
L0
M1
N2
O3
P4
Q5
R6"
trackable_list_wrapper
Á

-kernel
.bias
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

)kernel
*bias
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú_random_generator
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

+kernel
,bias
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í_random_generator
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
ô__call__
+õ&call_and_return_all_conditional_losses"
_tf_keras_layer
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
 "
trackable_list_wrapper
²
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Q
\0
]1
^2
_3
`4
a5
b6"
trackable_list_wrapper
«
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

/kernel
0bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

1kernel
2bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª_random_generator
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
«
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
<
/0
01
12
23"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
²
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Q
l0
m1
n2
o3
p4
q5
r6"
trackable_list_wrapper
«
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

3kernel
4bias
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô_random_generator
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

5kernel
6bias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç_random_generator
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layer
<
30
41
52
63"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
T
|0
}1
~2
3
4
5
6"
trackable_list_wrapper
«
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer
«
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

7kernel
8bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

9kernel
:bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤_random_generator
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
«
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
<
70
81
92
:3"
trackable_list_wrapper
<
70
81
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
X
0
1
2
3
4
5
6"
trackable_list_wrapper
«
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

;kernel
<bias
¾	variables
¿trainable_variables
Àregularization_losses
Á	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î_random_generator
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

=kernel
>bias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
«
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á_random_generator
â__call__
+ã&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
<
;0
<1
=2
>3"
trackable_list_wrapper
<
;0
<1
=2
>3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
X
0
1
2
3
 4
¡5
¢6"
trackable_list_wrapper
«
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
«
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

?kernel
@bias
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Akernel
Bbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
«
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
<
?0
@1
A2
B3"
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

±total

²count
³	variables
´	keras_api"
_tf_keras_metric
R

µtotal

¶count
·	variables
¸	keras_api"
_tf_keras_metric
c

¹total

ºcount
»
_fn_kwargs
¼	variables
½	keras_api"
_tf_keras_metric
c

¾total

¿count
À
_fn_kwargs
Á	variables
Â	keras_api"
_tf_keras_metric
c

Ãtotal

Äcount
Å
_fn_kwargs
Æ	variables
Ç	keras_api"
_tf_keras_metric
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_0_layer_call_fn_76511
*__inference_SDropout_0_layer_call_fn_76516´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76521
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76534´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_1_layer_call_fn_76539
*__inference_SDropout_1_layer_call_fn_76544´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76549
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76562´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
ð	variables
ñtrainable_variables
òregularization_losses
ô__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
_
L0
M1
N2
O3
P4
Q5
R6
J7
K8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_0_layer_call_fn_76567
*__inference_SDropout_0_layer_call_fn_76572´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76577
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76590´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_1_layer_call_fn_76595
*__inference_SDropout_1_layer_call_fn_76600´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76605
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76618´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
_
\0
]1
^2
_3
`4
a5
b6
Z7
[8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_0_layer_call_fn_76623
*__inference_SDropout_0_layer_call_fn_76628´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76633
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76646´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_1_layer_call_fn_76651
*__inference_SDropout_1_layer_call_fn_76656´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76661
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76674´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
_
l0
m1
n2
o3
p4
q5
r6
j7
k8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_0_layer_call_fn_76679
*__inference_SDropout_0_layer_call_fn_76684´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76689
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76702´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_1_layer_call_fn_76707
*__inference_SDropout_1_layer_call_fn_76712´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76717
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76730´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
b
|0
}1
~2
3
4
5
6
z7
{8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¾	variables
¿trainable_variables
Àregularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_0_layer_call_fn_76735
*__inference_SDropout_0_layer_call_fn_76740´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76745
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76758´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_1_layer_call_fn_76763
*__inference_SDropout_1_layer_call_fn_76768´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76773
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76786´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
h
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_0_layer_call_fn_76791
*__inference_SDropout_0_layer_call_fn_76796´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76801
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76814´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_SDropout_1_layer_call_fn_76819
*__inference_SDropout_1_layer_call_fn_76824´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76829
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76842´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
h
0
1
2
3
 4
¡5
¢6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
±0
²1"
trackable_list_wrapper
.
³	variables"
_generic_user_object
:  (2total
:  (2count
0
µ0
¶1"
trackable_list_wrapper
.
·	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¹0
º1"
trackable_list_wrapper
.
¼	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¾0
¿1"
trackable_list_wrapper
.
Á	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
.
Æ	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
#:!@2Adam/dense/kernel/m
:2Adam/dense/bias/m
?:=@2+Adam/tcn/residual_block_0/conv1D_0/kernel/m
5:3@2)Adam/tcn/residual_block_0/conv1D_0/bias/m
?:=@@2+Adam/tcn/residual_block_0/conv1D_1/kernel/m
5:3@2)Adam/tcn/residual_block_0/conv1D_1/bias/m
F:D@22Adam/tcn/residual_block_0/matching_conv1D/kernel/m
<::@20Adam/tcn/residual_block_0/matching_conv1D/bias/m
?:=@@2+Adam/tcn/residual_block_1/conv1D_0/kernel/m
5:3@2)Adam/tcn/residual_block_1/conv1D_0/bias/m
?:=@@2+Adam/tcn/residual_block_1/conv1D_1/kernel/m
5:3@2)Adam/tcn/residual_block_1/conv1D_1/bias/m
?:=@@2+Adam/tcn/residual_block_2/conv1D_0/kernel/m
5:3@2)Adam/tcn/residual_block_2/conv1D_0/bias/m
?:=@@2+Adam/tcn/residual_block_2/conv1D_1/kernel/m
5:3@2)Adam/tcn/residual_block_2/conv1D_1/bias/m
?:=@@2+Adam/tcn/residual_block_3/conv1D_0/kernel/m
5:3@2)Adam/tcn/residual_block_3/conv1D_0/bias/m
?:=@@2+Adam/tcn/residual_block_3/conv1D_1/kernel/m
5:3@2)Adam/tcn/residual_block_3/conv1D_1/bias/m
?:=@@2+Adam/tcn/residual_block_4/conv1D_0/kernel/m
5:3@2)Adam/tcn/residual_block_4/conv1D_0/bias/m
?:=@@2+Adam/tcn/residual_block_4/conv1D_1/kernel/m
5:3@2)Adam/tcn/residual_block_4/conv1D_1/bias/m
?:=@@2+Adam/tcn/residual_block_5/conv1D_0/kernel/m
5:3@2)Adam/tcn/residual_block_5/conv1D_0/bias/m
?:=@@2+Adam/tcn/residual_block_5/conv1D_1/kernel/m
5:3@2)Adam/tcn/residual_block_5/conv1D_1/bias/m
#:!@2Adam/dense/kernel/v
:2Adam/dense/bias/v
?:=@2+Adam/tcn/residual_block_0/conv1D_0/kernel/v
5:3@2)Adam/tcn/residual_block_0/conv1D_0/bias/v
?:=@@2+Adam/tcn/residual_block_0/conv1D_1/kernel/v
5:3@2)Adam/tcn/residual_block_0/conv1D_1/bias/v
F:D@22Adam/tcn/residual_block_0/matching_conv1D/kernel/v
<::@20Adam/tcn/residual_block_0/matching_conv1D/bias/v
?:=@@2+Adam/tcn/residual_block_1/conv1D_0/kernel/v
5:3@2)Adam/tcn/residual_block_1/conv1D_0/bias/v
?:=@@2+Adam/tcn/residual_block_1/conv1D_1/kernel/v
5:3@2)Adam/tcn/residual_block_1/conv1D_1/bias/v
?:=@@2+Adam/tcn/residual_block_2/conv1D_0/kernel/v
5:3@2)Adam/tcn/residual_block_2/conv1D_0/bias/v
?:=@@2+Adam/tcn/residual_block_2/conv1D_1/kernel/v
5:3@2)Adam/tcn/residual_block_2/conv1D_1/bias/v
?:=@@2+Adam/tcn/residual_block_3/conv1D_0/kernel/v
5:3@2)Adam/tcn/residual_block_3/conv1D_0/bias/v
?:=@@2+Adam/tcn/residual_block_3/conv1D_1/kernel/v
5:3@2)Adam/tcn/residual_block_3/conv1D_1/bias/v
?:=@@2+Adam/tcn/residual_block_4/conv1D_0/kernel/v
5:3@2)Adam/tcn/residual_block_4/conv1D_0/bias/v
?:=@@2+Adam/tcn/residual_block_4/conv1D_1/kernel/v
5:3@2)Adam/tcn/residual_block_4/conv1D_1/bias/v
?:=@@2+Adam/tcn/residual_block_5/conv1D_0/kernel/v
5:3@2)Adam/tcn/residual_block_5/conv1D_0/bias/v
?:=@@2+Adam/tcn/residual_block_5/conv1D_1/kernel/v
5:3@2)Adam/tcn/residual_block_5/conv1D_1/bias/vÒ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76521I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76534I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76577I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76590I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76633I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76646I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76689I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76702I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76745I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76758I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76801I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_0_layer_call_and_return_conditional_losses_76814I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ©
*__inference_SDropout_0_layer_call_fn_76511{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76516{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76567{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76572{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76623{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76628{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76679{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76684{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76735{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76740{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76791{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_0_layer_call_fn_76796{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76549I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76562I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76605I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76618I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76661I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76674I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76717I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76730I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76773I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76786I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76829I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_SDropout_1_layer_call_and_return_conditional_losses_76842I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ©
*__inference_SDropout_1_layer_call_fn_76539{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76544{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76595{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76600{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76651{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76656{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76707{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76712{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76763{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76768{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76819{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_SDropout_1_layer_call_fn_76824{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 __inference__wrapped_model_72899)*+,-./0123456789:;<=>?@AB6¢3
,¢)
'$
	tcn_inputÿÿÿÿÿÿÿÿÿ0
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_76506\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_layer_call_fn_76496O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÏ
E__inference_sequential_layer_call_and_return_conditional_losses_74588)*+,-./0123456789:;<=>?@AB>¢;
4¢1
'$
	tcn_inputÿÿÿÿÿÿÿÿÿ0
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
E__inference_sequential_layer_call_and_return_conditional_losses_74650)*+,-./0123456789:;<=>?@AB>¢;
4¢1
'$
	tcn_inputÿÿÿÿÿÿÿÿÿ0
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
E__inference_sequential_layer_call_and_return_conditional_losses_75116)*+,-./0123456789:;<=>?@AB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ0
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
E__inference_sequential_layer_call_and_return_conditional_losses_75550)*+,-./0123456789:;<=>?@AB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ0
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
*__inference_sequential_layer_call_fn_73728x)*+,-./0123456789:;<=>?@AB>¢;
4¢1
'$
	tcn_inputÿÿÿÿÿÿÿÿÿ0
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¦
*__inference_sequential_layer_call_fn_74526x)*+,-./0123456789:;<=>?@AB>¢;
4¢1
'$
	tcn_inputÿÿÿÿÿÿÿÿÿ0
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
*__inference_sequential_layer_call_fn_74717u)*+,-./0123456789:;<=>?@AB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ0
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
*__inference_sequential_layer_call_fn_74778u)*+,-./0123456789:;<=>?@AB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ0
p

 
ª "ÿÿÿÿÿÿÿÿÿº
#__inference_signature_wrapper_75613)*+,-./0123456789:;<=>?@ABC¢@
¢ 
9ª6
4
	tcn_input'$
	tcn_inputÿÿÿÿÿÿÿÿÿ0"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ¾
>__inference_tcn_layer_call_and_return_conditional_losses_76059|)*+,-./0123456789:;<=>?@AB7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¾
>__inference_tcn_layer_call_and_return_conditional_losses_76487|)*+,-./0123456789:;<=>?@AB7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
#__inference_tcn_layer_call_fn_75670o)*+,-./0123456789:;<=>?@AB7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "ÿÿÿÿÿÿÿÿÿ@
#__inference_tcn_layer_call_fn_75727o)*+,-./0123456789:;<=>?@AB7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "ÿÿÿÿÿÿÿÿÿ@