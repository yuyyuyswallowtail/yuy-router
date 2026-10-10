package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type parser struct {
	s   string
	pos int
}

func (p *parser) peek() byte {
	for p.pos < len(p.s) && p.s[p.pos] == ' ' {
		p.pos++
	}
	if p.pos >= len(p.s) {
		return 0
	}
	return p.s[p.pos]
}

// expr = term { (+|-) term }
func (p *parser) expr() (float64, error) {
	v, err := p.term()
	if err != nil {
		return 0, err
	}
	for {
		c := p.peek()
		if c != '+' && c != '-' {
			return v, nil
		}
		p.pos++
		r, err := p.term()
		if err != nil {
			return 0, err
		}
		if c == '+' {
			v += r
		} else {
			v -= r
		}
	}
}

// term = unary { (*|/|%) unary }
func (p *parser) term() (float64, error) {
	v, err := p.unary()
	if err != nil {
		return 0, err
	}
	for {
		c := p.peek()
		if c != '*' && c != '/' && c != '%' {
			return v, nil
		}
		p.pos++
		r, err := p.unary()
		if err != nil {
			return 0, err
		}
		switch c {
		case '*':
			v *= r
		case '/':
			if r == 0 {
				return 0, errors.New("division by zero")
			}
			v /= r
		case '%':
			if r == 0 {
				return 0, errors.New("modulo by zero")
			}
			v = math.Mod(v, r)
		}
	}
}

// unary = (+|-) unary | power
func (p *parser) unary() (float64, error) {
	switch p.peek() {
	case '-':
		p.pos++
		v, err := p.unary()
		return -v, err
	case '+':
		p.pos++
		return p.unary()
	}
	return p.power()
}

// power = atom [ ^ unary ]  (right-associative)
func (p *parser) power() (float64, error) {
	base, err := p.atom()
	if err != nil {
		return 0, err
	}
	if p.peek() == '^' {
		p.pos++
		exp, err := p.unary()
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

// atom = number | ( expr )
func (p *parser) atom() (float64, error) {
	c := p.peek()
	if c == '(' {
		p.pos++
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, errors.New("missing closing parenthesis )")
		}
		p.pos++
		return v, nil
	}
	start := p.pos
	for p.pos < len(p.s) && (p.s[p.pos] >= '0' && p.s[p.pos] <= '9' || p.s[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		if c == 0 {
			return 0, errors.New("incomplete expression")
		}
		return 0, fmt.Errorf("unknown character: %q", string(c))
	}
	return strconv.ParseFloat(p.s[start:p.pos], 64)
}

func main() {
	input := strings.TrimSpace(strings.Join(os.Args[1:], " "))
	if input == "" {
		fmt.Fprintln(os.Stderr, "Usage: go run calc.go \"(2 + 3) * 4 ^ 2 / 5 % 3\"")
		fmt.Fprintln(os.Stderr, "Operators: + - * / % ^ and parentheses")
		os.Exit(1)
	}

	p := &parser{s: strings.ReplaceAll(input, "**", "^")}
	v, err := p.expr()
	if err == nil && p.peek() != 0 {
		err = fmt.Errorf("unknown character: %q", string(p.s[p.pos]))
	}
	if err == nil && (math.IsInf(v, 0) || math.IsNaN(v)) {
		err = errors.New("result is undefined or too large")
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	if math.Abs(v) < 1e15 {
		v = math.Round(v*1e10) / 1e10 // drop rounding noise such as 0.30000000000000004
	}
	fmt.Println(input, "=", strconv.FormatFloat(v, 'f', -1, 64))
}
