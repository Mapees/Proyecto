#! /bin/bash
#cat out |sed -e 's/^/ /g' -e 's/ [^#][^ ]*//g' -e 's/^ *//g'
#sort out2 | uniq -c | sort -nr >> out3

grep -e '#irony' train.data > irony.data
grep -e '#sarcasm' train.data > sarcasm.data
grep -e  '#not' train.data > not.data
grep -v -i '#irony' train.data | grep -v -i '#sarcasm' | grep -v -i '#not' >> otros.data

cat irony.data | awk '{print $1}' >> irony_labels
cat sarcasm.data | awk '{print $1}' >> sarcasm_labels
cat not.data | awk '{print $1}' >> not_labels
cat otros.data | awk '{print $1}' >> otros_labels
